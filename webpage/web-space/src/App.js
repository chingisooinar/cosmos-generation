import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import Button from 'react-bootstrap/Button';

function App() {
  const [predUrl, setPredUrl] = useState();


  const handleSubmit = event => {

  const requestOptions = {
      method: 'GET',
       mode: 'no-cors',
      //headers: { 'Content-Type': 'multipart/form-data' }, // DO NOT INCLUDE HEADERS
  };
    fetch('http://127.0.0.1/generate')
    .then(response => response.json())
    .then(data => {
      console.log(data)
        setPredUrl("data:image/png;base64,"+data[0]);
    })
    .catch((err) => console.log(err));
  }
  return (  <div className="d-flex">
      <div className="card mx-5">
        <div className="mx-auto mt-5">
          <Button onClick={handleSubmit}>
          Predict
          </Button>
        </div>
      </div>
      <div className="card">
        <div className="d-flex mx-auto">
          <h1 className="mt-2">Random Space Image Generation</h1>
        </div>

          <div className="image-container m-4">
            <img src={predUrl} width="256" height="256" alt="" />
            <h3>Generated Image</h3>
          </div>
        </div>

      </div>
);
}

export default App;
