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
    <div className="min-h-screen bg-gray-950 flex items-center justify-center overflow-hidden">
      
      {/* Background accent */}
      <div className="absolute w-96 h-96 bg-red-900 opacity-20 rounded-full blur-3xl top-10 left-10" />
      <div className="absolute w-80 h-80 bg-red-800 opacity-10 rounded-full blur-3xl bottom-10 right-10" />

      {/* Card utama */}
      <div className={`relative z-10 flex flex-col items-center gap-6 transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
        
         <div className="bg-gray-400 border border-gray-700 rounded-2xl p-10 shadow-xl justify-center items-center flex flex-col gap-2">
            <img src={logo} alt="Telkom University" className="w-28 h-28 object-contain" />
            <h1 className="text-white text-3xl font-bold tracking-wide">
            Face Recognition
          </h1>
          <div className="w-16 h-1 bg-red-600 mx-auto my-3 rounded-full" />
          <h2 className="text-red-600 text-lg font-bold text-shadow-black-500">
            Access Gate System
          </h2>
          <p className="text-gray-900 text-sm leading-relaxed">
            Sistem autentikasi berbasis pengenalan wajah untuk akses gate parkir motor Universitas Telkom
          </p>
            </div>

        {/* Tombol */}
        <button
          onClick={() => navigate('/login')}
          className="mt-2 px-10 py-3 bg-red-700 hover:bg-red-600 text-white font-semibold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-red-900 active:scale-95"
        >
          Masuk ke Sistem →
        </button>

      </div>
    </div>
  )
}

export default WelcomePage