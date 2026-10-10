import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error)
      return (
        <main className="fatal">
          <h1>BioMaker</h1>
          <p>界面遇到问题。重新加载后，已提交的实验仍会保留。</p>
          <p>
            The interface encountered a problem. Submitted experiments are
            saved.
          </p>
          <pre>{this.state.error.message}</pre>
          <button
            className="primary-button"
            onClick={() => window.location.reload()}
          >
            重新加载 / Reload
          </button>
        </main>
      );
    return this.props.children;
  }
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
