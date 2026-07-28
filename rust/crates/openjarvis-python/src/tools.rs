//! PyO3 bindings for tool types.

use openjarvis_tools::traits::BaseTool;
use pyo3::prelude::*;
use std::sync::Arc;

fn security_disabled(tool_name: &str) -> PyResult<String> {
    Err(PyErr::new::<pyo3::exceptions::PyPermissionError, _>(
        format!(
            "Direct '{tool_name}' binding disabled: use an authenticated ToolExecutor \
             with explicit capability grants and live confirmation"
        ),
    ))
}

#[pyclass(name = "ToolExecutor")]
pub struct PyToolExecutor {
    pub inner: Arc<openjarvis_tools::ToolExecutor>,
}

#[pymethods]
impl PyToolExecutor {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(openjarvis_tools::ToolExecutor::new(None, None)),
        }
    }

    fn list_tools(&self) -> Vec<String> {
        self.inner.list_tools()
    }

    fn execute(&self, tool_name: &str, params_json: &str) -> PyResult<String> {
        let params: serde_json::Value = serde_json::from_str(params_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let result = self
            .inner
            .execute(tool_name, &params, None, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(serde_json::to_string(&result).unwrap_or_default())
    }
}

#[pyclass(name = "CalculatorTool")]
pub struct PyCalculatorTool;

#[pymethods]
impl PyCalculatorTool {
    #[new]
    fn new() -> Self {
        Self
    }

    fn execute(&self, expression: &str) -> PyResult<String> {
        let tool = openjarvis_tools::builtin::calculator::CalculatorTool;
        let params = serde_json::json!({"expression": expression});
        let result = tool
            .execute(&params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.content)
    }
}

#[pyclass(name = "ThinkTool")]
pub struct PyThinkTool;

#[pymethods]
impl PyThinkTool {
    #[new]
    fn new() -> Self {
        Self
    }

    fn execute(&self, thought: &str) -> PyResult<String> {
        let tool = openjarvis_tools::builtin::think::ThinkTool;
        let params = serde_json::json!({"thought": thought});
        let result = tool
            .execute(&params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.content)
    }
}

#[pyclass(name = "FileReadTool")]
pub struct PyFileReadTool;

#[pymethods]
impl PyFileReadTool {
    #[new]
    fn new() -> Self {
        Self
    }

    fn execute(&self, path: &str) -> PyResult<String> {
        let _ = path;
        security_disabled("file_read")
    }
}

#[pyclass(name = "FileWriteTool")]
pub struct PyFileWriteTool;

#[pymethods]
impl PyFileWriteTool {
    #[new]
    fn new() -> Self {
        Self
    }

    fn execute(&self, path: &str, content: &str) -> PyResult<String> {
        let _ = (path, content);
        security_disabled("file_write")
    }
}

#[pyclass(name = "ShellExecTool")]
pub struct PyShellExecTool;

#[pymethods]
impl PyShellExecTool {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (command, cwd=None))]
    fn execute(&self, command: &str, cwd: Option<&str>) -> PyResult<String> {
        let _ = (command, cwd);
        security_disabled("shell_exec")
    }
}

#[pyclass(name = "HttpRequestTool")]
pub struct PyHttpRequestTool;

#[pymethods]
impl PyHttpRequestTool {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (url, method="GET", body=None))]
    fn execute(&self, url: &str, method: &str, body: Option<&str>) -> PyResult<String> {
        let _ = (url, method, body);
        security_disabled("http_request")
    }
}

#[pyclass(name = "GitStatusTool")]
pub struct PyGitStatusTool;

#[pymethods]
impl PyGitStatusTool {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (cwd=None))]
    fn execute(&self, cwd: Option<&str>) -> PyResult<String> {
        let _ = cwd;
        security_disabled("git_status")
    }
}

#[pyclass(name = "GitDiffTool")]
pub struct PyGitDiffTool;

#[pymethods]
impl PyGitDiffTool {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (cwd=None))]
    fn execute(&self, cwd: Option<&str>) -> PyResult<String> {
        let _ = cwd;
        security_disabled("git_diff")
    }
}

#[pyclass(name = "GitLogTool")]
pub struct PyGitLogTool;

#[pymethods]
impl PyGitLogTool {
    #[new]
    fn new() -> Self {
        Self
    }

    #[pyo3(signature = (cwd=None, count=None))]
    fn execute(&self, cwd: Option<&str>, count: Option<u32>) -> PyResult<String> {
        let _ = (cwd, count);
        security_disabled("git_log")
    }
}
