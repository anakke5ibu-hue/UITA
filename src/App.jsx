import { BrowserRouter, Routes, Route } from 'react-router-dom'
import WelcomePage from './pages/WelcomePage'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import DisplayGatePage from './pages/DisplayGatePage'
import RegistrasiPage from './pages/RegistrasiPage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/registrasi" element={<RegistrasiPage />} />
        <Route path="/gate" element={<DisplayGatePage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App