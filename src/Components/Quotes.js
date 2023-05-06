import React,{useState,useEffect} from "react"

function Quotes() {
    const [text,setQuote] = useState('');
    const [author,setAuthor]= useState('');

    useEffect(()=>{
        getQuote();
       const intervalID = setInterval(()=>{
        getQuote()
       }, 24 * 60 * 60);
    return ()=>{
        clearInterval(intervalID);
    }
    },[])
    function getQuote() {
        fetch('https://type.fit/api/quotes')
        .then(res=> res.json())
        .then(data=>{
            console.log(data);
            setQuote(data[Math.floor((Math.random() * 1600) + 1)].text);
            setAuthor(data[Math.floor((Math.random() * 1600) + 1)].author);
        })
    }
return(
    <div className="qoute-section">
    <p className="qoute-text">"{text}"</p>
    <p className="qoute-author">-{author}</p>
    </div>
)
}
export default Quotes;
