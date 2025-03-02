import React, { use, useEffect, useState } from "react";
import axios from "axios";
import { Chart, registerables } from "chart.js";
import Sidebar from "./components/Navbar";

Chart.register(...registerables);

const App = () => {
    const [data, setData] = useState(null);

    useEffect(() => {
      async function fetchData() {
        const response = await axios.get("http://127.0.0.1:5000/api/data");
        setData(response.data);
      }
      fetchData();
    }
    , []);


  if (!data) return <h2>Loading...</h2>;

  console.log(data.conf_matrix);

  return (
      <div id="app-container-id" className="flex h-screen">
        {/* Sidebar (Left) */}
        <div className="w-64 bg-gray-100 border-r ">
          <Sidebar />
        </div>
    
        {/* Main Content (Right) */}
        <div className="flex-1 flex flex-col items-center justify-center p-6">
          <h1 className="text-2xl font-bold text-center">Fatality Predictor Dashboard</h1>
    
          {/* Confusion Matrix Section */}
          <div className="mt-6 text-center">
            <h2 className="text-xl font-semibold">Confusion Matrix</h2>
            <img src={data.conf_matrix} alt="Confusion Matrix" className="mt-4 w-auto max-w-full" />
            <img src={data.scatter_plot} alt="Scatter Plot" className="mt-4 w-auto max-w-full" />
            <img src={data.pie_chart} alt="Scatter Plot" className="mt-4 w-auto max-w-full" />

          </div>
        </div>
      </div>
 );
    


};

export default App;
