//! ToolExecutor — central dispatch with RBAC, taint, timeout.

use crate::builtin::BuiltinTool;
use crate::traits::BaseTool;
use openjarvis_core::error::{OpenJarvisError, ToolError};
use openjarvis_core::{EventBus, EventType, ToolResult};
use openjarvis_security::capabilities::CapabilityPolicy;
use openjarvis_security::taint::{TaintSet, check_taint};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub struct ToolExecutor {
    tools: HashMap<String, BuiltinTool>,
    capability_policy: Option<CapabilityPolicy>,
    bus: Option<Arc<EventBus>>,
    default_timeout: Duration,
}

impl ToolExecutor {
    pub fn new(
        capability_policy: Option<CapabilityPolicy>,
        bus: Option<Arc<EventBus>>,
    ) -> Self {
        Self {
            tools: HashMap::new(),
            capability_policy,
            bus,
            default_timeout: Duration::from_secs(30),
        }
    }

    pub fn register(&mut self, tool: BuiltinTool) {
        let id = tool.tool_id().to_string();
        self.tools.insert(id, tool);
    }

    pub fn get_tool(&self, name: &str) -> Option<&BuiltinTool> {
        self.tools.get(name)
    }

    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn tool_specs(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.to_openai_function()).collect()
    }

    pub fn execute(
        &self,
        tool_name: &str,
        params: &Value,
        agent_id: Option<&str>,
        taint: Option<&TaintSet>,
    ) -> Result<ToolResult, OpenJarvisError> {
        let tool = self.tools.get(tool_name).ok_or_else(|| {
            OpenJarvisError::Tool(ToolError::NotFound(tool_name.to_string()))
        })?;

        // A tool call without both an explicit policy and a non-empty runtime
        // principal is denied. Missing security context is never an allow-all
        // compatibility mode.
        let policy = self.capability_policy.as_ref().ok_or_else(|| {
            OpenJarvisError::Tool(ToolError::CapabilityDenied(
                "<missing-policy>".to_string(),
                format!("tool:invoke (tool: {tool_name})"),
            ))
        })?;
        let aid = agent_id.filter(|value| !value.trim().is_empty()).ok_or_else(|| {
            OpenJarvisError::Tool(ToolError::CapabilityDenied(
                "<missing-identity>".to_string(),
                format!("tool:invoke (tool: {tool_name})"),
            ))
        })?;
        let resource = authorization_resource(tool_name, params);
        let spec = tool.spec();
        if !policy.check(aid, "tool:invoke", &resource) {
            return Err(OpenJarvisError::Tool(ToolError::CapabilityDenied(
                aid.to_string(),
                format!("tool:invoke (resource: {resource})"),
            )));
        }
        for cap in &spec.required_capabilities {
            if !policy.check(aid, cap, &resource) {
                return Err(OpenJarvisError::Tool(ToolError::CapabilityDenied(
                    aid.to_string(),
                    format!("{cap} (resource: {resource})"),
                )));
            }
        }

        // Taint check
        if let Some(taint_set) = taint {
            if let Some(violation) = check_taint(tool_name, taint_set) {
                return Err(OpenJarvisError::Tool(ToolError::TaintViolation(
                    tool_name.to_string(),
                    violation,
                )));
            }
        }

        // The Rust executor has no authenticated live-confirmation callback.
        // Sensitive built-ins must remain unavailable instead of treating
        // construction or an RPC call as implicit consent.
        if spec.requires_confirmation {
            return Err(OpenJarvisError::Tool(ToolError::ConfirmationRequired(
                tool_name.to_string(),
            )));
        }

        // Emit start event
        if let Some(ref bus) = self.bus {
            let mut data = HashMap::new();
            data.insert("tool_name".to_string(), Value::String(tool_name.to_string()));
            bus.publish(EventType::ToolCallStart, data);
        }

        let start = std::time::Instant::now();
        let timeout = Duration::from_secs_f64(tool.spec().timeout_seconds);
        let timeout = if timeout.is_zero() { self.default_timeout } else { timeout };

        let mut result = tool.execute(params)?;
        let elapsed = start.elapsed();

        if elapsed > timeout {
            if let Some(ref bus) = self.bus {
                let mut data = HashMap::new();
                data.insert("tool_name".to_string(), Value::String(tool_name.to_string()));
                data.insert(
                    "outcome".to_string(),
                    Value::String("completed_after_deadline".to_string()),
                );
                bus.publish(EventType::ToolTimeout, data);
            }
            // Execution is synchronous here, so by the time the deadline can
            // be observed the action has already completed. Report that truth
            // rather than claiming a timeout cancelled the side effect.
            result.metadata.insert(
                "deadline_exceeded".to_string(),
                Value::Bool(true),
            );
            result.metadata.insert(
                "outcome".to_string(),
                Value::String("completed_after_deadline".to_string()),
            );
        }

        // Emit end event
        if let Some(ref bus) = self.bus {
            let mut data = HashMap::new();
            data.insert("tool_name".to_string(), Value::String(tool_name.to_string()));
            data.insert("duration_seconds".to_string(), Value::Number(
                serde_json::Number::from_f64(elapsed.as_secs_f64()).unwrap(),
            ));
            bus.publish(EventType::ToolCallEnd, data);
        }

        Ok(result)
    }
}

fn authorization_resource(tool_name: &str, params: &Value) -> String {
    const RESOURCE_KEYS: [&str; 17] = [
        "path",
        "file_path",
        "destination",
        "dest",
        "url",
        "uri",
        "endpoint",
        "workspace",
        "repo",
        "cwd",
        "device_id",
        "device",
        "entity_id",
        "entity",
        "channel",
        "to",
        "action",
    ];
    for key in RESOURCE_KEYS {
        let Some(value) = params.get(key) else {
            continue;
        };
        let raw = match value {
            Value::String(text) => text.trim().to_string(),
            Value::Number(number) => number.to_string(),
            _ => continue,
        };
        if raw.is_empty() {
            continue;
        }
        if matches!(key, "path" | "file_path" | "cwd") {
            return canonical_resource_path(&raw);
        }
        return raw;
    }
    format!("tool:{tool_name}")
}

fn canonical_resource_path(raw: &str) -> String {
    let path = Path::new(raw);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    if let Ok(canonical) = std::fs::canonicalize(&absolute) {
        return canonical.to_string_lossy().into_owned();
    }
    if let (Some(parent), Some(file_name)) = (absolute.parent(), absolute.file_name()) {
        if let Ok(canonical_parent) = std::fs::canonicalize(parent) {
            return canonical_parent
                .join(file_name)
                .to_string_lossy()
                .into_owned();
        }
    }
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjarvis_security::capabilities::CapabilityPolicy;

    #[test]
    fn test_executor_register_and_execute() {
        let mut policy = CapabilityPolicy::default();
        policy.grant("test-agent", "tool:invoke", "tool:calculator");
        let mut exec = ToolExecutor::new(Some(policy), None);
        exec.register(BuiltinTool::Calculator(crate::builtin::calculator::CalculatorTool));
        let result = exec
            .execute(
                "calculator",
                &serde_json::json!({"expression": "2+2"}),
                Some("test-agent"),
                None,
            )
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_executor_missing_policy_fails_closed() {
        let mut exec = ToolExecutor::new(None, None);
        exec.register(BuiltinTool::Calculator(crate::builtin::calculator::CalculatorTool));
        let err = exec
            .execute(
                "calculator",
                &serde_json::json!({"expression": "2+2"}),
                Some("test-agent"),
                None,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OpenJarvisError::Tool(ToolError::CapabilityDenied(_, _))
        ));
    }

    #[test]
    fn test_executor_missing_identity_fails_closed() {
        let mut policy = CapabilityPolicy::default();
        policy.grant("test-agent", "tool:invoke", "tool:calculator");
        let mut exec = ToolExecutor::new(Some(policy), None);
        exec.register(BuiltinTool::Calculator(crate::builtin::calculator::CalculatorTool));
        let err = exec
            .execute(
                "calculator",
                &serde_json::json!({"expression": "2+2"}),
                None,
                None,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OpenJarvisError::Tool(ToolError::CapabilityDenied(_, _))
        ));
    }

    #[test]
    fn test_executor_sensitive_tool_requires_confirmation() {
        let mut policy = CapabilityPolicy::default();
        policy.grant("test-agent", "tool:invoke", "*");
        policy.grant("test-agent", "code:execute", "*");
        let mut exec = ToolExecutor::new(Some(policy), None);
        exec.register(BuiltinTool::ShellExec(crate::builtin::shell::ShellExecTool));
        let err = exec
            .execute(
                "shell_exec",
                &serde_json::json!({"command": "echo must-not-run"}),
                Some("test-agent"),
                None,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            OpenJarvisError::Tool(ToolError::ConfirmationRequired(_))
        ));
    }

    #[test]
    fn test_authorization_resource_normalizes_traversal() {
        let resource = authorization_resource(
            "file_write",
            &serde_json::json!({"path": "/safe/../etc/passwd"}),
        );
        assert_eq!(resource, "/etc/passwd");
    }

    #[test]
    fn test_authorization_resource_normalizes_cwd_traversal() {
        let resource = authorization_resource(
            "git_status",
            &serde_json::json!({"cwd": "/safe/../etc"}),
        );
        assert_eq!(resource, "/etc");
    }

    #[test]
    fn test_executor_tool_not_found() {
        let exec = ToolExecutor::new(None, None);
        let err = exec
            .execute("nonexistent", &serde_json::json!({}), None, None)
            .unwrap_err();
        assert!(matches!(err, OpenJarvisError::Tool(ToolError::NotFound(_))));
    }
}
