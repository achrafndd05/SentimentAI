import React, { useState } from 'react';
import './App.css';
import axios from 'axios';
import './index.js';
import { Link } from 'react-router-dom';
const App = () => {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [percentage, setPercentage] = useState('');

 
const backendURL = 'http://127.0.0.1:5000'; // Replace with your backend URL
const api = axios.create({
  baseURL: backendURL,
});
  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleProcess = async () => {
    try {
      // const response = await axios.post('/process', { text: inputText });


      const response = await fetch('http://127.0.0.1:5000/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text:inputText })
                });
      
      // const { type, percentage } = response.data
      // const  type  = response.data


      const data = await response.json();
      setOutputText(data.result);
      // setOutputText(type)
      // setPercentage('90%')
    } catch (error) {
      console.error(error);
    }
    // Perform processing on the input text and set the output
    // Here, we'll just copy the input text as an example
  };

  const handleDelete = () => {
    setOutputText('');
    setPercentage('')
  };

  return (
    <div>
      <div className="topBar">
        <div className="websiteName">Sent.AI</div>
        <div className="pageTitle">Sentiment Analysis</div>
        <Link to="/">
            <button className="homehome">Home</button>
    </Link>
        <Link to="/twitter">
            <button className="second-page">Scrape in Twitter</button>
        </Link>
      
      </div>
      <div className="content">
        <div className="leftSide">
          <div>Add your text here</div>
          <input
            className="textInput"
            type="text"
            value={inputText}
            onChange={(event) => setInputText(event.target.value)}
            placeholder="Write the text you want to classify..."
          />
          {console.log(inputText)}
          <button className="button" onClick={handleProcess}>
            Classify
          </button>
        </div>
        <div className="rightSide">
          <div className="title">Results</div>
          <div className='titre'>
          <div className='tags'>TAG</div>
          <div className='percentage'>PERCENTAGE</div>
          </div>
          <div className='line'>__________________________________________________________________</div>
          <div className="output">
            {outputText}
            <div className='percentage-num'>{percentage}</div>
              <button className="deleteButton" onClick={handleDelete}>
                Clear
              </button>
            
          </div>
        </div>
      </div>
    </div>
  );
};
export default App;