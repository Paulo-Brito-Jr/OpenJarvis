//! RBAC capability system — fine-grained permission model for tool dispatch.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    #[serde(rename = "file:read")]
    FileRead,
    #[serde(rename = "file:write")]
    FileWrite,
    #[serde(rename = "network:fetch")]
    NetworkFetch,
    #[serde(rename = "code:execute")]
    CodeExecute,
    #[serde(rename = "memory:read")]
    MemoryRead,
    #[serde(rename = "memory:write")]
    MemoryWrite,
    #[serde(rename = "channel:send")]
    ChannelSend,
    #[serde(rename = "email:write")]
    EmailWrite,
    #[serde(rename = "calendar:write")]
    CalendarWrite,
    #[serde(rename = "message:send")]
    MessageSend,
    #[serde(rename = "approval:decide")]
    ApprovalDecide,
    #[serde(rename = "tool:invoke")]
    ToolInvoke,
    #[serde(rename = "schedule:create")]
    ScheduleCreate,
    #[serde(rename = "system:admin")]
    SystemAdmin,
}

impl Capability {
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::FileRead => "file:read",
            Capability::FileWrite => "file:write",
            Capability::NetworkFetch => "network:fetch",
            Capability::CodeExecute => "code:execute",
            Capability::MemoryRead => "memory:read",
            Capability::MemoryWrite => "memory:write",
            Capability::ChannelSend => "channel:send",
            Capability::EmailWrite => "email:write",
            Capability::CalendarWrite => "calendar:write",
            Capability::MessageSend => "message:send",
            Capability::ApprovalDecide => "approval:decide",
            Capability::ToolInvoke => "tool:invoke",
            Capability::ScheduleCreate => "schedule:create",
            Capability::SystemAdmin => "system:admin",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CapabilityGrant {
    pub capability: String,
    pub pattern: String,
}

#[derive(Debug, Clone)]
struct AgentPolicy {
    grants: Vec<CapabilityGrant>,
    deny: Vec<String>,
}

/// RBAC capability policy for tool dispatch.
///
/// Policies should be deny-by-default. Callers can explicitly construct
/// `CapabilityPolicy::new(false)` only for legacy compatibility.
pub struct CapabilityPolicy {
    policies: HashMap<String, AgentPolicy>,
    default_deny: bool,
}

impl CapabilityPolicy {
    pub fn new(default_deny: bool) -> Self {
        Self {
            policies: HashMap::new(),
            default_deny,
        }
    }

    pub fn grant(&mut self, agent_id: &str, capability: &str, pattern: &str) {
        let policy = self.policies.entry(agent_id.to_string()).or_insert_with(|| {
            AgentPolicy {
                grants: Vec::new(),
                deny: Vec::new(),
            }
        });
        policy.grants.push(CapabilityGrant {
            capability: capability.to_string(),
            pattern: pattern.to_string(),
        });
    }

    pub fn deny(&mut self, agent_id: &str, capability: &str) {
        let policy = self.policies.entry(agent_id.to_string()).or_insert_with(|| {
            AgentPolicy {
                grants: Vec::new(),
                deny: Vec::new(),
            }
        });
        policy.deny.push(capability.to_string());
    }

    pub fn check(&self, agent_id: &str, capability: &str, resource: &str) -> bool {
        let policy = match self.policies.get(agent_id) {
            Some(p) => p,
            None => return !self.default_deny,
        };

        // A malformed policy must never become broader merely because one
        // backend ignores syntax that another backend understands.
        if policy.deny.iter().any(|pattern| !valid_glob_pattern(pattern))
            || policy.grants.iter().any(|grant| {
                !valid_glob_pattern(&grant.capability)
                    || !valid_glob_pattern(&grant.pattern)
            })
        {
            return false;
        }

        for denied in &policy.deny {
            if glob_match(denied, capability) {
                return false;
            }
        }

        for grant in &policy.grants {
            if glob_match(&grant.capability, capability) {
                if grant.pattern == "*" {
                    return true;
                }
                if !resource.is_empty() && glob_match(&grant.pattern, resource) {
                    return true;
                }
            }
        }

        !self.default_deny
    }

    pub fn list_agents(&self) -> Vec<String> {
        self.policies.keys().cloned().collect()
    }

    pub fn load_json(&mut self, json_str: &str) -> Result<(), serde_json::Error> {
        let data: serde_json::Value = serde_json::from_str(json_str)?;
        if let Some(agents) = data["agents"].as_array() {
            for agent_data in agents {
                let agent_id = agent_data["agent_id"].as_str().unwrap_or("");
                if agent_id.is_empty() {
                    continue;
                }
                if let Some(grants) = agent_data["grants"].as_array() {
                    for g in grants {
                        let cap = g["capability"].as_str().unwrap_or("");
                        let pat = g["pattern"].as_str().unwrap_or("*");
                        self.grant(agent_id, cap, pat);
                    }
                }
                if let Some(deny_list) = agent_data["deny"].as_array() {
                    for d in deny_list {
                        if let Some(cap) = d.as_str() {
                            self.deny(agent_id, cap);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for CapabilityPolicy {
    fn default() -> Self {
        Self::new(true)
    }
}

fn valid_glob_pattern(pattern: &str) -> bool {
    !pattern.is_empty() && !pattern.chars().any(|ch| matches!(ch, '?' | '[' | ']'))
}

fn glob_match(pattern: &str, text: &str) -> bool {
    if !valid_glob_pattern(pattern) {
        return false;
    }
    if !pattern.contains('*') {
        return pattern == text;
    }

    let parts: Vec<&str> = pattern.split('*').collect();
    let mut position = 0;

    let first = parts.first().copied().unwrap_or("");
    if !first.is_empty() {
        if !text.starts_with(first) {
            return false;
        }
        position = first.len();
    }

    for part in parts.iter().skip(1).take(parts.len().saturating_sub(2)) {
        if part.is_empty() {
            continue;
        }
        if let Some(found) = text[position..].find(part) {
            position += found + part.len();
        } else {
            return false;
        }
    }

    let last = parts.last().copied().unwrap_or("");
    if !last.is_empty() {
        let Some(suffix_start) = text.len().checked_sub(last.len()) else {
            return false;
        };
        if suffix_start < position || !text.ends_with(last) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_allow() {
        let policy = CapabilityPolicy::new(false);
        assert!(policy.check("agent1", "file:read", ""));
    }

    #[test]
    fn test_default_deny() {
        let policy = CapabilityPolicy::new(true);
        assert!(!policy.check("agent1", "file:read", ""));
    }

    #[test]
    fn test_default_trait_is_deny_by_default() {
        let policy = CapabilityPolicy::default();
        assert!(!policy.check("agent1", "file:read", ""));
    }

    #[test]
    fn test_explicit_grant() {
        let mut policy = CapabilityPolicy::new(true);
        policy.grant("agent1", "file:read", "*");
        assert!(policy.check("agent1", "file:read", ""));
        assert!(!policy.check("agent1", "file:write", ""));
    }

    #[test]
    fn test_scoped_grant_requires_nonempty_matching_resource() {
        let mut policy = CapabilityPolicy::new(true);
        policy.grant("agent1", "file:read", "/safe/*");
        assert!(policy.check("agent1", "file:read", "/safe/data.txt"));
        assert!(!policy.check("agent1", "file:read", "/etc/passwd"));
        assert!(!policy.check("agent1", "file:read", ""));
    }

    #[test]
    fn test_explicit_deny_overrides_grant() {
        let mut policy = CapabilityPolicy::new(false);
        policy.grant("agent1", "file:*", "*");
        policy.deny("agent1", "file:write");
        assert!(policy.check("agent1", "file:read", ""));
        assert!(!policy.check("agent1", "file:write", ""));
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("file:*", "file:read"));
        assert!(!glob_match("file:read", "file:write"));
        assert!(glob_match("*.txt", "doc.txt"));
        assert!(glob_match("a*b*c", "a--b--c"));
        assert!(!glob_match("a*a", "a"));
        assert!(!glob_match("file:?", "file:r"));
        assert!(!glob_match("file:[rw]", "file:r"));
        assert!(!glob_match("", ""));
    }

    #[test]
    fn malformed_policy_patterns_deny_entire_policy() {
        let mut malformed_grant = CapabilityPolicy::new(false);
        malformed_grant.grant("agent1", "file:?", "*");
        assert!(!malformed_grant.check("agent1", "file:read", ""));

        let mut malformed_deny = CapabilityPolicy::new(false);
        malformed_deny.deny("agent1", "code:[a-z]*");
        assert!(!malformed_deny.check("agent1", "file:read", ""));
    }
}
