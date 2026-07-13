import { useState, useEffect } from 'react'  // ← import useEffect juga
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [visible, setVisible] = useState(false)   // state buat animasi muncul

  useEffect(() => {
    setTimeout(() => setVisible(true), 100)       // muncul setelah 100ms
  }, [])

  const handleLogin = () => {
    if (username === 'admin' && password === 'admin123') {
      localStorage.setItem('isLoggedIn', 'true')
      navigate('/dashboard')
    } else {
      alert('Incorrect username or password!')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-red-950 to-gray-400 flex items-center justify-center overflow-hidden relative">
      
      {/* Background accent - pakai animasi float juga (opsional) */}
      <div 
        className="absolute w-96 h-96 bg-yellow-400 opacity-20 rounded-full blur-3xl top-10 left-10"
        style={{ animation: 'float1 1.5s ease-in-out infinite' }}
      />
      <div 
        className="absolute w-80 h-96 bg-white opacity-20 rounded-full blur-3xl bottom-10 left-10"
        style={{ animation: 'float2 2s ease-in-out infinite' }}
      />
      <div 
        className="absolute w-96 h-80 bg-yellow-500 opacity-20 rounded-full blur-3xl bottom-30 right-10"
        style={{ animation: 'float1 1.5s ease-in-out infinite reverse' }}
      />
      <div 
        className="absolute w-80 h-80 bg-white opacity-20 rounded-full blur-3xl top-5 right-10"
        style={{ animation: 'float2 2s ease-in-out infinite alternate' }}
      />

      {/* Card Login dengan animasi naik */}
      <div 
        className={`relative z-10 flex flex-col items-center gap-6 transition-all duration-700 ${
          visible ? 'opacity-75 translate-y-0' : 'opacity-0 translate-y-10'
        }`}
      >
        <div className="bg-white outline-red-950 outline-3 rounded-2xl shadow-2xl p-10 w-full max-w-md flex flex-col items-center gap-6">
            <button
              onClick={() => navigate('/')}
              className="self-start flex items-center gap-2 text-gray-800 hover:text-gray-300 transition-all duration-200 hover:-translate-x-1 mb-2"
            >
              ⮜ Back
            </button>
          <img src={logo} alt="Telkom University" className="w-20 h-20 object-contain" />
          <h2 className="text-2xl font-bold text-gray-800">Face Recognition Access Gate</h2>
          <p className="text-red-900 text-md font-semibold">Operator Login</p>

          {/* FORM */}
          <div className="w-full flex flex-col gap-4">
            <div>
              <label className="text-gray-800 text-sm mb-1 block">SSO Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Masukkan username"
                className="w-full bg-gray-300 text-black rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600 placeholder-gray-500"
              />
            </div>
            <div>
              <label className="text-gray-800 text-sm mb-1 block">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Masukkan password"
                className="w-full bg-gray-300 text-black rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600 placeholder-gray-500"
              />
            </div>
          </div>

          {/* Tombol Login */}
          <button
            onClick={handleLogin}
            className="w-full bg-red-700 hover:bg-red-600 text-white font-semibold py-3 rounded-xl transition-all duration-300 hover:scale-105 active:scale-95"
          >
            Login
          </button>

        </div>
      </div>
    </div>
  )
}

export default LoginPage