// Dynamic lifecycle menu mirrors prototype/system/examples/liveportrait/pinokio.js.
module.exports = {
  version: "8.0.0",
  title: "Roop 3D",
  description: "Independent local Roop 3D workspace",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    for (const [script, text] of [["install.js", "Installing Roop 3D"], ["update.js", "Updating Roop 3D"], ["reset.js", "Resetting Roop 3D environment"]]) {
      if (info.running(script)) return [{ default: true, icon: "fa-solid fa-terminal", text, href: script }]
    }
    if (info.running("start.js")) {
      const local = info.local("start.js")
      if (local && /^http:\/\/127\.0\.0\.1:\d+$/.test(local.url || "")) {
        return [{ default: true, icon: "fa-solid fa-cube", text: "Open Roop 3D", href: local.url },
          { icon: "fa-solid fa-terminal", text: "Roop 3D terminal", href: "start.js" }]
      }
      return [{ default: true, icon: "fa-solid fa-terminal", text: "Starting Roop 3D", href: "start.js" }]
    }
    const installed = info.exists("cache/install-ready.json")
    return [{ default: true, icon: installed ? "fa-solid fa-play" : "fa-solid fa-plug",
      text: installed ? "Start Roop 3D" : "Install Roop 3D", href: installed ? "start.js" : "install.js" },
      { icon: "fa-solid fa-arrows-rotate", text: "Update Roop 3D", href: "update.js" },
      { icon: "fa-solid fa-wrench", text: "Reinstall environment", href: "install.js" },
      { icon: "fa-solid fa-rotate-left", text: "Reset environment", href: "reset.js",
        confirm: "Reset only Roop 3D's dedicated Python environment? Source, outputs, models and Roop Ultimate are preserved." }]
  }
}
