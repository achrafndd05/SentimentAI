import React ,{useState, useEffect}from "react";
import Chart from 'react-apexcharts';
import './App.css'
function Donutchart({series})
{ 
    const chartcolors =['#00e396','#e52f2f','#008ffb']

    return(
        <React.Fragment>
            <div className='chart'>        
            <Chart 
            type="donut"
            width={400}
            height={500}
            series={series}

            options={{
             labels:['Positive','Negative','Neutral'],
             title:{
               // align:"center",
             },
             plotOptions:{
             pie:{
                donut:{
                    labels:{
                        show:true,
                        total:{
                            show:true,
                            showAlways:true,
                             //formatter: () => '343',
                            fontSize:20,
                            color: '#feb019',
                        }
                    }
                }
             }
             },
             dataLabels:{
                enabled:true,
             },
             colors : chartcolors,
            }}
            />
            </div>
        </React.Fragment>
    );
}
export default Donutchart;