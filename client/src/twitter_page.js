import React, { useState } from "react";
import './App.css';
import { BrowserRouter as Router, Link, Route, Switch } from 'react-router-dom';

const TwitterPage = () => {
  const [hashtag, setHashtag] = useState("");
  const [number, setNumber] = useState(0);
  const [outputText, setOutputText] = useState('');
  const handleHashtagChange = (event) => {
    setHashtag(event.target.value);
  };

  const handleNumberChange = (event) => {
    setNumber(parseInt(event.target.value));
  };
  const handleProcess = async () => {
    try {
      // const response = await axios.post('/process', { text: inputText });
      // const response = await fetch('http://127.0.0.1:5000/process', {
      //               method: 'POST',
      //               headers: {
      //                   'Content-Type': 'application/json'
      //               },
      //               body: JSON.stringify({ hashtag, number })
      //           });
    
      // // const { type, percentage } = response.data
      // // const  type  = response.data
      // const data = await response.json();
      setOutputText('Positive');
      // setOutputText(type)
      // setNumber(data.proba)
    } catch (error) {
      console.error(error);
    }
  };
  const handleDelete = () => {
    setOutputText('');
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
            value={number}
            onChange={handleNumberChange}
          />
        </label>
  <label className="hashlabel">
          Hashtag:
          <input
          className="hashtag"
            type="text"
            value={hashtag}
            onChange={handleHashtagChange}
            placeholder = 'Enter your hashtag'
          />
        </label> 
        <button className="buttonanalysis" onClick={handleProcess}>
            Run analysis
          </button>
  </div>
  <div className="analysis">
    <div className="result">Results</div>
    <div className="output>">{outputText}</div>
  </div>
  <button className="clean" onClick={handleDelete}>Clear</button>
</div>
  );
}

export default TwitterPage;
