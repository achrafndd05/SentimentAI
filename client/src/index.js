import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import TwitterPage from './twitter_page';
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
const router = createBrowserRouter([
  {
    path: "/",
    element: <App/>,
  },
  {
    path: "twitter",
    element: <TwitterPage/>,
  },
]);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <RouterProvider router={router}/>
);

