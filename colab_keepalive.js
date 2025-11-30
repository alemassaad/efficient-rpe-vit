// Paste this into browser console (Cmd+Option+J on Mac, Ctrl+Shift+J on Windows)
// Keeps Colab session alive by clicking connect button every 60 seconds

function keepAlive() {
    console.log("Keeping alive " + new Date());
    document.querySelector("colab-connect-button").click();
}
setInterval(keepAlive, 60000);
