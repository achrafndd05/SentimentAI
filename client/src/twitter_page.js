import React, { useState } from "react";
import './App.css';
import { BrowserRouter as Router, Link, Route, Switch, HashRouter } from 'react-router-dom';
import Donutchart from "./chart";
const TwitterPage = () => {
  const [hashtag, setHashtag] = useState("");
  const [number, setNumber] = useState(0);
  const [posnum, setPosNum] = useState('');
  const [negnum, setNegNum] = useState('');
  const [netnum, setNetNum] = useState('');
  const [chartData,setChartData]= useState([])

  const handleHashtagChange = (event) => {
    setHashtag(event.target.value);
  };

  const handleNumberChange = (event) => {
    setNumber(parseInt(event.target.value));
  };
  const handleProcess = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/process_hashtag', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ hashtag:hashtag, number:number})
                });
    
      
      const data = await response.json();
      setPosNum(data.pos);
      setNegNum(data.neg);
      setNetNum(data.net);
      
    } catch (error) {
      console.error(error);
    }
  };
  const handleDelete = () => {
      setPosNum('');
      setNegNum('');
      setNetNum('');
      setChartData([]);
      
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
    <div className="buttons">
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
         <div className="but">
       <div> <button className="buttonanalysis" onClick={handleProcess}>
            Run analysis
          </button></div>
          <div><button className="clean" onClick={handleDelete}>Clear</button></div>
          </div>
          </div>
  </div>
  <div className="analysis">
    <div className="stat">
    <div className="result">Results</div>
    <div className="cont">
    <div className="tagg">Tag</div>
    <div className="tweet">Tweets</div>
    </div>
    <div className='linee'>___________________________________</div>
    
    <br/>
    <div className="cont">
      <div className="Sentp">Positive : </div>
      <div className="outputt">{posnum}</div>
    </div>
    <div className='linee'>___________________________________</div>
    <div className="contt">
      <div className="Sentng">Negative : </div>
      <div className="outputt">{negnum}</div>
    </div>
    <div className='linee'>___________________________________</div>
    <div className="contt">
      <div className="Sentn">Neutral : </div>
      <div className="outputt">{netnum}</div>
    </div>
    </div>
    {posnum !== '' && negnum !== '' && netnum !== '' && (
         <div className="chart"><Donutchart series={[parseInt(posnum), parseInt(negnum), parseInt(netnum)]} /></div>
         )}
         </div>
           
</div>
  );
}

export default TwitterPage;
