import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
// import logo from './img/logo.png';
import './App.css';
import AppStart from './index/start/appStart';

function Home() {
  const navigate = useNavigate();
  return (
    <div>
      <button onClick={() => navigate('/appstart')}>app-start-lin</button>
    </div>
  );
}


function App() {

const logo = './img/logo.png';
// const navigateToAppStart = () => {
//   window.location.href = './index/start/index.html';
// };
const navigateToAppStart = () => {
  window.location.href = './index/start/appStart.tsx';
}

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        <a
          className="App-start-link"
          onClick={navigateToAppStart}
          // href="./start"
          target="_blank"
          rel="noopener noreferrer"
        >
          Get Start
        </a>
      </header>
    </div>
  );
}

export default App;
