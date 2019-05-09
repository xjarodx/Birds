function openFileLoad() {
    var x = document.getElementById("seabird");
    if (x.style.visibility === "hidden") {
      x.style.visibility = "visible";
    } else {
      x.style.visibility = "hidden";
    }
  } 
  
//var image =  document.getElementById("search");

//function gotIt() {
//  if (image.getAttribute('src') == "/static/html_Images/birdIconWithText.png") {
//      image.src = "/static/html_Images/Icon_Bird_512x512.png";}
//  else {
//      image.src = "/static/html_Images/birdIconWithText.png";}
//}

function gotIt() {
  var x = document.getElementById("wikipic");
  if (x.style.visibility === "hidden") {
    x.style.visibility = "visible";
  } else {
    x.style.visibility = "hidden";
  }
  var y = document.getElementById("wikitable");
  if (x.style.visibility === "hidden") {
    x.style.visibility = "visible";
  } else {
    x.style.visibility = "hidden";
  }
} 

//function redirect(){
//  window.location = "http://127.0.0.1:5000/";
//}