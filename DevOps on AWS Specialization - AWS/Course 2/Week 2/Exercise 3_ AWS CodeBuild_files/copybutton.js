function addCopyButtonToCode(){
    // get all code elements
    var allCodeBlocksElements = document.querySelectorAll("div.sourceCode pre");

    // For each element, do the following steps
    allCodeBlocksElements.forEach(function(currentValue, currentIndex, listObj) {
        // define a unique id for this element and add it
        var currentId = "codeblock" + currentIndex;
        currentValue.id = currentId;

        // create a button that's configured for clipboard.js
        // point it to the text that's in this code block
        var clipButton = document.createElement("button");
        clipButton.className = "btn copybtn";
        clipButton.setAttribute("data-clipboard-target", "#" + currentId);
        clipButton.innerHTML = '<img class="clipboard" src="images/clipboard.svg" width="13" alt="Copy to clipboard">'

        // add the button just after the text in the code block
        currentValue.parentNode.insertBefore(clipButton, currentValue.nextSibling);
    });

    // tell clipboard.js to look for clicks that match this query
    new Clipboard('.btn');
}

window.addEventListener('load', function(){
    addCopyButtonToCode();
});
