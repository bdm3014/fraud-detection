const tmpResultMsg = "please, enter form parameters and click PREDICT button to get the result...";

read_json();

function read_json() {
   const bgReqColor = "rgb(231 233 184)";

   fetch(window.location.protocol + "//" + window.location.host + '/data/fields.json').then(res => res.json()).then(fields => {
         
         const HTMLResponse = document.getElementById("new_input_div");
         
         // table
         let tblInput = document.createElement("table");
         tblInput.className = "table table-striped table-bordered table-sm";
         tblInput.cellspacing = "0";
         tblInput.width = "100%";
         HTMLResponse.appendChild(tblInput);
         
         //head
         let thead = document.createElement("thead");
         tblInput.appendChild(thead);

         let trHead = document.createElement("tr");
         thead.appendChild(trHead);

         let thColFeat = document.createElement("th");
         thColFeat.textContent = "Feature";
         thColFeat.style = "background-color:lightgray;";
         trHead.appendChild(thColFeat);
         
         let thColValue = document.createElement("th");
         thColValue.textContent = "Value";
         thColValue.style = "background-color:lightgray;";
         trHead.appendChild(thColValue);
         
         //body
         let tBody = document.createElement("tbody");
         tblInput.appendChild(tBody);
         
         fields.forEach(field => {
            
            let trBody = document.createElement("tr");
            tBody.appendChild(trBody);

            let tdColFeat = document.createElement("td");
            tdColFeat.textContent = field.label + " " + (field.labelPlus ? field.labelPlus : "");
            trBody.appendChild(tdColFeat);
            
            let tdColCtrl = document.createElement("td");
            trBody.appendChild(tdColCtrl);
            
            switch (field.type) {
               case "text":
                    let textField = document.createElement("input");
                    textField.type = "text";
                    textField.id = field.name;
                    textField.name = field.name;
                    textField.label = field.label;
                    textField.className = "param textField";
                    textField.style.width = "200px";
                    
                    if (Boolean(field.required))
                    {
                       textField.required = true;
                       textField.style = "background-color:" + bgReqColor + "; width:200px;";  
                    };
                    
                    tdColCtrl.appendChild(textField);
                    break;
                    
               case "number":
                    let txtNumField = document.createElement("input");
                    txtNumField.type = "number";
                    txtNumField.subtype = "integer";
                    txtNumField.id = field.name;
                    txtNumField.name = field.name;
                    txtNumField.label = field.label;
                    txtNumField.min = field.min;
                    txtNumField.max = field.max;
                    txtNumField.step = field.step;
                    txtNumField.className = "param txtNumField";
                    txtNumField.style.width = "200px";

                    if (Boolean(field.required))
                    {
                       txtNumField.required = true;
                       txtNumField.style = "background-color:" + bgReqColor + "; width:200px;";
                    };

                    tdColCtrl.appendChild(txtNumField);
                    break;
                    
               case "decimal":
                    let txtDecField = document.createElement("input");
                    txtDecField.type = "number";
                    txtDecField.subtype = "decimal";
                    txtDecField.id = field.name;
                    txtDecField.name = field.name;
                    txtDecField.label = field.label;
                    txtDecField.min = field.min;
                    txtDecField.max = field.max;
                    txtDecField.step = field.step;
                    txtDecField.className = "param txtDecField";
                    txtDecField.style.width = "200px";

                    if (Boolean(field.required))
                    {
                       txtDecField.required = true;
                       txtDecField.style = "background-color:" + bgReqColor + "; width:200px;";
                    };

                    tdColCtrl.appendChild(txtDecField);
                    break;
                    
               case "list": 
                    let listField = document.createElement("select");
                    listField.id = field.name;
                    listField.name = field.name;
                    listField.label = field.label;
                    listField.className = "param listField";
                    listField.style.width = "200px";
                    
                    if (Boolean(field.required))
                    {
                         listField.required = true;
                         listField.style = "background-color:" + bgReqColor + "; width:200px;";
                    }
                    
                    tdColCtrl.appendChild(listField);
                    
                    for (i in field.values) {
                        let optField = document.createElement("option");
                        optField.value = field.values[i].code;
                        optField.textContent = field.values[i].value; 
                        listField.appendChild(optField);
                    }

                    listField.value = "";
                    
                    break;
                    
               case "radio":
                    
                    for (i in field.values) {
                        let rdField = document.createElement("input");
                        rdField.type = "radio";
                        rdField.name = field.name;
                        rdField.id = field.values[i].code;
                        rdField.value = field.values[i].code;
                        rdField.label = field.label;
                        rdField.className = "param rdField";
                      //rdField.textContent = field.values[i].value;
                        tdColCtrl.appendChild(rdField);

                        let lblField = document.createElement("label");
                        lblField.textContent = field.values[i].value;
                        tdColCtrl.appendChild(lblField);

                        tdColCtrl.appendChild(document.createElement("br"));
                    }
                    
                    break;
                    
               case "datetime":
                    let dateTimeField = document.createElement("input");
                    dateTimeField.type = "datetime-local";
                    dateTimeField.id = field.name;
                    dateTimeField.name = field.name;
                    dateTimeField.label = field.label;
                    dateTimeField.min = field.min;
                    dateTimeField.max = field.max;
                  //dateTimeField.value = currentDateTime();
                    dateTimeField.className = "param dateTimeField";
                    dateTimeField.style.width = "200px";
                    
                    if (Boolean(field.required))
                    {
                       dateTimeField.required = true;
                       dateTimeField.style = "background-color:" + bgReqColor + "; width:200px;";
                    };
                    
                    tdColCtrl.appendChild(dateTimeField);
                    
                    break;
            };
 
       });
   });

 }

 function validateFields()
 {
     const paramList = document.getElementsByClassName("param");
     
     for (i in paramList) {
         switch (paramList[i].type) {
            case "text":
            case "select-one":
            case "datetime-local":
                 
                 if (paramList[i].required)
                 {
                    if (paramList[i].value == "")
                    {
                       alert(paramList[i].label + " is required!");
                       paramList[i].focus();
                       return;              
                    };
                 };
                 
                 break;
                 
            case "number":
                  
                 if (paramList[i].value == "")
                 {
                    if (paramList[i].required)
                    {
                       alert(paramList[i].label + " is required!");
                       paramList[i].focus();
                       return;
                    };
                 }
                 else
                 {
                    if (paramList[i].subtype == "integer")
                    {
                       if (paramList[i].value.indexOf(".") > -1 || paramList[i].value.indexOf(",") > -1)
                       {
                          alert(paramList[i].label + " must be an integer value");
                          paramList[i].focus();
                          return;
                       };
                    };
                       
                    const valField = parseFloat(paramList[i].value);
                    const minField = parseFloat(paramList[i].min);
                    const maxField = parseFloat(paramList[i].max);
                       
                    if (valField < minField || valField > maxField)
                    {
                       alert(paramList[i].label + " must be between " + minField + " and " + maxField);
                       paramList[i].focus();
                       return;
                    };

                 };
                 
                 break;
                                  
            case "radio":
                 break;
         };
     };

     getPrediction();
 }

 function getPrediction()
 {
   // defining API rest URL
   const API_URL = window.location.protocol + "//" + window.location.host + "/api/prediction";

   // initializing the label result
   const lblResult = document.getElementById("lblResult");
   lblResult.innerHTML   = "<br><strong>PREDICTING...</strong>";

   // reading input params
   const paramList = document.getElementsByClassName("param");
   let params = {};
   
   for (i in paramList) {
       switch (paramList[i].type) {
          case "text":
          case "number":
          case "decimal":
          case "select-one":
          case "datetime-local":
               //console.log(paramList[i].name + " : ", paramList[i].value);
               params[paramList[i].name] = paramList[i].value;
               break;
          case "radio":
               if (paramList[i].checked) {
                  //console.log(paramList[i].name + " : ", paramList[i].value);
                  params[paramList[i].name] = paramList[i].value;
               };
               break;
       };
   };
   
   // making up the API request
   let requestHead = new Headers();
   requestHead.append("Content-Type", "application/json");
   
   let requestOpt = {}; 
   requestOpt.method = "POST";
   requestOpt.body = JSON.stringify(params);
   requestOpt.headers = requestHead;
   
   fetch(`${API_URL}`, requestOpt).then(res => res.json()).catch(error => console.error('Error:', error)).then(response => {
        const result = response.result;
        lblResult.innerHTML  = "<br><strong>" + result + "</strong>";
   });

 }

 function reset()
 {
     const lblResult = document.getElementById("lblResult");
     lblResult.innerHTML = "<br><strong>" + tmpResultMsg + "</strong>";
     
     const paramList = document.getElementsByClassName("param");
     
     for (i in paramList) {
         switch (paramList[i].type) {
            case "text":
            case "number":
            case "decimal":
            case "select-one":
            case "datetime-local":
                 document.getElementById(paramList[i].id).value = "";
                 break;
            case "radio":
                 document.getElementById(paramList[i].id).checked = false;
                 break;
         };
     };

 }

function recent_predictions()
{
   window.open("/index.html", "_self");
}

function load_examples(pTipo)
{
   const lblResult = document.getElementById("lblResult");
   lblResult.innerHTML = "<br><strong>" + tmpResultMsg + "</strong>";

   fetch(window.location.protocol + "//" + window.location.host + '/data/fields.json').then(res => res.json()).then(fields => {
      fields.forEach(field => {
         const inputField = document.getElementById(field.name);
         inputField.value = field[pTipo];
      });
   });
}

function currentDateTime()
{
    dateNow       = new Date();
    const year    = dateNow.getFullYear();
    const month   = dateNow.getMonth() + 1;
    const day     = dateNow.getDate();
    const hour    = dateNow.getHours();
    const minute  = dateNow.getMinutes();
    const dateStr = year + "-" + ("0" + month).slice(-2) + "-" + ("0" + day).slice(-2) + "T" + ("0" + hour).slice(-2) + ":" + ("0" + minute).slice(-2);
    
    return dateStr;
 }
