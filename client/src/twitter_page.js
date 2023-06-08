import React, { useState } from "react";
import './App.css';
import { BrowserRouter as Router, Link, Route, Switch } from 'react-router-dom';

const TwitterPage = () => {
  const [textInput, setTextInput] = useState("");
  const [numberInput, setNumberInput] = useState(0);

  const handleTextInputChange = (event) => {
    setTextInput(event.target.value);
  };

  const handleNumberInputChange = (event) => {
    setNumberInput(parseInt(event.target.value));
  };

  return (
    <div>
      <div className="topBar">
        <div className="websiteName">Sent.AI</div>
        <div className="pageTitle">Sentiment Analysis</div>
        <Link to="/">
          <button className="home">Home</button>
        </Link>
        <Link to="/twitter">
          <button className="second-pagetwit">Scrape in Twitter</button>
        </Link>
      </div>

  <div>
  <label className="numlabel">
          Number of Tweets:
          <input
          className="num-tweet"
            type="number"
            value={numberInput}
            onChange={handleNumberInputChange}
          />
        </label>
  <label className="hashlabel">
          Hashtag:
          <input
          className="hashtag"
            type="text"
            value={textInput}
            onChange={handleTextInputChange}
            placeholder = 'Enter your hashtag'
          />
        </label>
        
  </div>
</div>
  );
}

export default TwitterPage;
