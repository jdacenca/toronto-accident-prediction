import { useState } from "react";
import { Home, BarChart, Share2, Package, Star, FileText, MapPin } from "lucide-react";
import "../styles/Navbar.css";

const Navbar = () => {
  const [active, setActive] = useState("Dashboard");

  const menuItems = [
    { name: "Dashboard", icon: <Home size={18} /> },
    { name: "Data Insights", icon: <BarChart size={18} /> },
    { name: "Models", icon: <Share2 size={18} /> },
    { name: "Performance Insights", icon: <Package size={18} /> },
    { name: "Best Model", icon: <Star size={18} /> },
    { name: "Predictions", icon: <FileText size={18} /> },
    { name: "Map", icon: <MapPin size={18} /> },
  ];

  return (
    <div className="h-screen w-64 bg-gray-50 border-r p-5 flex flex-col">
      <div className="flex items-center space-x-2 mb-8">
        <div className="w-6 h-6 bg-violet-500 text-white flex items-center justify-center rounded-full">
          <span className="text-sm font-bold"></span>
        </div>
        <h1 className="text-lg font-semibold text-violet-600">Fatality Predictor</h1>
      </div>
      <nav className="flex flex-col space-y-4">
        {menuItems.map((item, index) => (
          <div
            key={index}
            className={`flex items-center space-x-3 px-4 py-2 rounded-lg cursor-pointer hover:bg-gray-100 transition-all text-gray-500 ${
              active === item.name ? "text-violet-600 font-semibold" : ""
            }`}
            onClick={() => setActive(item.name)}
          >
            {item.icon}
            <span>{item.name}</span>
          </div>
        ))}
      </nav>
    </div>
  );
};

export default Navbar;
