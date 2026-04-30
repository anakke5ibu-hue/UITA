import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleLogin = () => {
    if (username === 'admin' && password === 'admin123') {
      navigate('/dashboard')
    } else {
      alert('Username atau password salah!')
    }
  }

  return (
    // KOTAK LUAR = background layar penuh
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-red-950 to-gray-900 flex items-center justify-center">
      
      {/* KOTAK DALAM = card login */}
      <div className="bg-gray-400 rounded-2xl shadow-2xl p-10 w-full max-w-md flex flex-col items-center gap-6">

        {/* TOMBOL BACK - paling atas, rata kiri */}
        <button
          onClick={() => navigate('/')}
          className="self-start flex items-center gap-2 text-gray-800 hover:text-white transition-all duration-200 hover:-translate-x-1"
        >
          ← Kembali
        </button>

        {/* LOGO */}
        <img src={logo} alt="Telkom University" className="w-20 h-20 object-contain" />
        
        {/* JUDUL */}
        <div className="text-center">
          <h1 className="text-black text-2xl font-bold">Face Recognition Access Gate</h1>
          <p className="text-gray-800 text-sm mt-1">Operator Login</p>
        </div>

        {/* FORM */}
        <div className="w-full flex flex-col gap-4">
          <div>
            <label className="text-gray-800 text-sm mb-1 block">SSO Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Masukkan username"
              className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600 placeholder-gray-500"
            />
          </div>
          <div>
            <label className="text-gray-800 text-sm mb-1 block">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Masukkan password"
              className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-600 placeholder-gray-500"
            />
          </div>
        </div>

        {/* TOMBOL LOGIN */}
        <button
          onClick={handleLogin}
          className="w-full bg-red-700 hover:bg-red-600 text-white font-semibold py-3 rounded-xl transition-all duration-300 hover:scale-105 shadow-lg"
        >
          Login SSO
        </button>

      </div>
      {/* TUTUP KOTAK DALAM */}

    </div>
    // TUTUP KOTAK LUAR
  )
}

export default LoginPage