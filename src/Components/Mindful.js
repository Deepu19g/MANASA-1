import React from "react";
import Navbar from "./Navbar.js";
import Footer from "./Footer.js";
import { DefaultContext } from "react-icons";
import "../App.css";

const Mindful = () => {
    return (
        <div  ><Navbar/><hr></hr>
         <div className="title" >
         <div className="exercise" style={{align:"center"}} >
            <img src=""></img>
            <h1>1. Think about what actions make you feel better</h1>
            <p>These may range from activities that take five minutes to several days. Create a shortlist of things and carry it with your day. When you sense a need to improve your wellbeing in this area, your list allows you to quickly opt into something that is feasible at this moment. Even though you may not be able to go for a three-day hike, you can always make an investment in yourself in smaller ways.</p>
         </div>
         <div className="exercise" style={{align:"center"}}
         >
            <img src=""></img>
            <h1>2. Take an active role in your wellbeing</h1>
            <p>Know that you are more than a passenger, you're an active driver of your mental, emotional and physical well-being. Make plans for when to include self-care in your daily routines. Schedule time in your calendar, or attach a new habit to an old one – such as brushing your teeth can be a trigger for a small self-care action every day. While spending time on self-care will not prevent you from coming across difficult periods in your life, you will be better prepared to manage them.</p>
         </div>
         <div className="exercise" style={{align:"center"}}
         >
            <img src=""></img>
            <h1>3. Small actions matter</h1>
            <p>Even if you just take one minute to take ten mindful breaths counts. From a neuropsychology point of view, you are training your brain to tell yourself that you are worthy of this time.</p>
         </div>
         <div className="exercise" style={{align:"center"}}
         >
            <img src=""></img>
            <h1>4. Spending time with loved ones</h1>
            <p>Whether going for a walk with a friend or family member or simply spending time talking and hanging out together, social interaction can help reduce stress levels and promote positive emotions.</p>
         </div>
         </div>
         <Footer/>
         </div>
    );
};

export default Mindful;