import { useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import logo from '../assets/logo-telkom.png'

function WelcomePage() {
  const navigate = useNavigate()
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    setTimeout(() => setVisible(true), 100)
  }, [])
  
    
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-400 via-red-950 to-black flex items-center justify-center overflow-hidden">
      
      {/* Background accent - dengan animasi */}
<div 
  className="absolute w-96 h-96 bg-white opacity-20 rounded-full blur-3xl top-10 left-10"
  style={{ animation: 'float1 1.5s ease-in-out infinite' }}
/>
<div 
  className="absolute w-80 h-96 bg-yellow-400 opacity-20 rounded-full blur-3xl bottom-10 left-10"
  style={{ animation: 'float2 2s ease-in-out infinite' }}
/>
<div 
  className="absolute w-96 h-80 bg-white opacity-10 rounded-full blur-3xl bottom-10 right-10"
  style={{ animation: 'float1 1.5s ease-in-out infinite reverse' }}
/>
<div 
  className="absolute w-80 h-80 bg-yellow-400 opacity-20 rounded-full blur-3xl top-10 right-10"
  style={{ animation: 'float2 2s ease-in-out infinite alternate' }}
/>
      {/* Card utama */}
      <div className={`relative z-10 flex flex-col items-center gap-6 transition-all duration-700 ${visible ? 'opacity-60 translate-y-0' : 'opacity-0 translate-y-10'}`}>
        
         <div className="bg-white outline-red-950 outline-3 rounded-2xl p-10 shadow-xl justify-center items-center flex flex-col gap-2">
            <img src={logo} alt="Telkom University" className="w-28 h-28 object-contain" />
            <h1 className="text-gray-900 text-3xl font-bold tracking-wide">
            Face Recognition
          </h1>
          <div className="w-50 h-1 bg-red-900 mx-auto my-2 rounded-full" />
          <h2 className="text-red-600 text-lg font-bold text-shadow-black-500">
            Access Gate System
          </h2>
          <p className="text-gray-900 text-sm leading-relaxed">
            Face Recognition-Based Authentication System for Motorcycle Parking Gate Access at Telkom University
          </p>
            </div>

        {/* Tombol */}
        <button
          onClick={() => navigate('/login')}
          className="mt-2 px-10 py-3 bg-red-700 hover:bg-red-600 text-white font-semibold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-red-900 active:scale-95"
        >
          Enter the System
        </button>

      </div>
    </div>
  )
}

export default WelcomePage