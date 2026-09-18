import React, { useState } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import ToolView from './ToolView.jsx'
import './index.css'

/* Two views over the same backend. "tool" is the Phase 7 deliverable: register
 * any two files and take the artifact set away. "study" is the Phase 2
 * instrument study it came out of, kept because its numbers are the evidence
 * for the method the tool selects. */
function Root() {
  const [view, setView] = useState('tool')
  return view === 'tool'
    ? <ToolView view={view} setView={setView} />
    : <App view={view} setView={setView} />
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)
