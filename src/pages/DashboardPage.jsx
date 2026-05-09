import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

// ═══════════════════════════════════════════════════════════════
// KONFIGURASI SERVER
// ═══════════════════════════════════════════════════════════════
const SERVER_URL = 'http://localhost:8001'

// ─── Simpan 1 log ke tabel log_akses ─────────────────────────
const saveLog = async (entry) => {
  if (entry.status !== 'GRANTED') return
  try {
    const res = await fetch(`${SERVER_URL}/log_akses`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        nama: entry.nama,
        nim: entry.nim,
        waktu: entry.waktu,
        tanggal: entry.tanggal,
        gate: entry.gate,
        confidence: entry.confidence || null,
        status: entry.status,
      }),
    })
    const text = await res.text()
    console.log('Server response:', res.status, text)
  } catch (e) {
    console.error('Gagal simpan log:', e)
  }
}

// ─── Ambil semua log dari server ─────────────────────────────
const fetchLogs = async () => {
  try {
    const res = await fetch(`${SERVER_URL}/log_akses`)
    const data = await res.json()
    if (!Array.isArray(data)) return []
    return data.map((r) => {
      const tanggalStr = r.tanggal ? String(r.tanggal).substring(0, 10) : toDateStr(new Date())
      return {
        id: r.no,
        nama: r.nama,
        nim: r.nim,
        waktu: r.waktu,
        tanggal: tanggalStr,
        timestamp: new Date(tanggalStr + 'T00:00:00'),
        gate: r.gate,
        confidence: r.confidence,
        status: r.status,
      }
    })
  } catch (e) {
    console.error('Gagal ambil log:', e)
    return []
  }
}

// ─── Ambil semua user dari tabel users_parkir ────────────────
const fetchUsers = async () => {
  try {
    const res = await fetch(`${SERVER_URL}/users`)
    const data = await res.json()
    if (!Array.isArray(data)) return []
    return data
  } catch (e) {
    console.error('Gagal ambil users:', e)
    return []
  }
}

// ─── Blokir user: hapus dari users_parkir ────────────────────
const blockUser = async (nim) => {
  try {
    const res = await fetch(`${SERVER_URL}/users/${encodeURIComponent(nim)}/block`, {
      method: 'POST',
    })
    return res.ok
  } catch (e) {
    console.error('Gagal blokir:', e)
    return false
  }
}

const unblockUser = async (nim) => {
  try {
    const res = await fetch(`${SERVER_URL}/users/${encodeURIComponent(nim)}/unblock`, {
      method: 'POST',
    })
    return res.ok
  } catch (e) {
    console.error('Gagal unblock:', e)
    return false
  }
}

// ─── Utilitas Tanggal ─────────────────────────────────────────
const toDateStr = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

const getFilterRange = (filter) => {
  const now = new Date()
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  switch (filter) {
    case 'hari_ini':
      return { start: today, end: new Date(today.getTime() + 86400000) }
    case 'kemarin': {
      const k = new Date(today.getTime() - 86400000)
      return { start: k, end: today }
    }
    case '2_hari_lalu': {
      const d = new Date(today.getTime() - 2 * 86400000)
      return { start: d, end: new Date(today.getTime() - 86400000) }
    }
    case '1_bulan': {
      const m = new Date(today)
      m.setMonth(m.getMonth() - 1)
      return { start: m, end: new Date(today.getTime() + 86400000) }
    }
    default:
      return null
  }
}

// ─── Export Excel ─────────────────────────────────────────────
let sheetJsLoaded = false
const loadSheetJs = () =>
  new Promise((resolve) => {
    if (window.XLSX) { resolve(window.XLSX); return }
    if (sheetJsLoaded) {
      const check = setInterval(() => {
        if (window.XLSX) { clearInterval(check); resolve(window.XLSX) }
      }, 100)
      return
    }
    sheetJsLoaded = true
    const s = document.createElement('script')
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js'
    s.onload = () => resolve(window.XLSX)
    document.head.appendChild(s)
  })

const exportToExcel = async (logs, filterLabel) => {
  const XLSX = await loadSheetJs()
  const rows = logs.map((l) => ({
    Nama: l.nama,
    NIM: l.nim,
    Waktu: l.waktu,
    Tanggal: l.tanggal,
    Gate: l.gate || 'Gate 4',
    Status: l.status,
    'Confidence (%)': l.confidence ? (parseFloat(l.confidence) * 100).toFixed(2) : '-',
  }))
  const ws = XLSX.utils.json_to_sheet(rows)
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'Log Akses')
  XLSX.writeFile(wb, `log-akses-${filterLabel.replace(/ /g, '-')}-${toDateStr(new Date())}.xlsx`)
}

// ─── Badge Status ─────────────────────────────────────────────
const StatusBadge = ({ status }) => (
  <span className={`text-xs font-bold px-3 py-1 rounded-full tracking-wide ${
    status === 'GRANTED'
      ? 'bg-green-900/60 text-green-400 border border-green-500/40'
      : 'bg-red-900/60 text-red-400 border border-red-500/40'
  }`}>
    {status}
  </span>
)

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
function DashboardPage() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('log')
  const [logs, setLogs] = useState([])
  const [registeredUsers, setRegisteredUsers] = useState([])
  const [filter, setFilter] = useState('hari_ini')
  const [searchQuery, setSearchQuery] = useState('')
  const [isConnected, setIsConnected] = useState(false)
  const [wsStatus, setWsStatus] = useState('Menghubungkan...')
  const [exporting, setExporting] = useState(false)
  const [loadingLogs, setLoadingLogs] = useState(true)
  const [loadingBlokir, setLoadingBlokir] = useState(true)
  const [blokirAction, setBlokirAction] = useState(null)
  const [blokirSearchQuery, setBlokirSearchQuery] = useState('')
  const [stats, setStats] = useState({ granted: 0, denied: 0, total: 0 })

  const wsRef = useRef(null)
  const nimCooldownRef = useRef({})
  const COOLDOWN_MS = 10000

  // ─── Load data awal dari server ──────────────────────────────
  useEffect(() => {
    // Ambil log history dari server
    fetchLogs().then((data) => {
      setLogs(data)
      setStats({
        granted: data.filter(l => l.status === 'GRANTED').length,
        denied: data.filter(l => l.status === 'DENIED').length,
        total: data.length,
      })
      setLoadingLogs(false)
    })

    // Ambil daftar user terdaftar dari users_parkir
    fetchUsers().then((data) => {
      setRegisteredUsers(data)
      setLoadingBlokir(false)
    })
  }, [])

  // ─── WebSocket: simpan realtime ke server ────────────────────
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/detect')
      wsRef.current = ws

      ws.onopen = () => { setIsConnected(true); setWsStatus('Terhubung') }  
      ws.onclose = () => {
        setIsConnected(false)
        setWsStatus('Terputus — mencoba ulang...')
        setTimeout(connect, 3000)
      }
      ws.onerror = () => setWsStatus('Error koneksi')

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          const currentStatus = data.status

          if (currentStatus !== 'granted' && currentStatus !== 'denied') return
          if (!data.faces?.length) return

          const isGranted = currentStatus === 'granted'
          const faceNim = isGranted ? (data.user?.nim || 'UNKNOWN') : (data.faces?.[0]?.nim || 'UNKNOWN')
          const cooldownKey = faceNim
          const now = Date.now()
          const lastTime = nimCooldownRef.current[cooldownKey] || 0

          if (now - lastTime < COOLDOWN_MS) return

          nimCooldownRef.current[cooldownKey] = now
          const dateNow = new Date()
          const newEntry = {
            id: now,
            nama: isGranted ? data.user?.nama || 'Tidak Dikenali' : 'Tidak Dikenali',
            nim: isGranted ? data.user?.nim || '-' : '-',
            status: isGranted ? 'GRANTED' : 'DENIED',
            waktu: dateNow.toLocaleTimeString('id-ID'),
            tanggal: toDateStr(dateNow),
            timestamp: dateNow,
            gate: 'Gate 4',
            confidence: isGranted ? data.user?.confidence || 0 : 0,
          }

          // Simpan ke server (background)
          saveLog(newEntry)

          // Update tampilan langsung tanpa tunggu Supabase
          setLogs((prev) => [newEntry, ...prev])
          setStats((prev) => ({
            granted: prev.granted + (isGranted ? 1 : 0),
            denied: prev.denied + (!isGranted ? 1 : 0),
            total: prev.total + 1,
          }))
        } catch (e) {
          console.error('WS parse error:', e)
        }
      }
    }

    connect()
    return () => { if (wsRef.current) wsRef.current.close() }
  }, [])

  // ─── Filter ───────────────────────────────────────────────────
  const filteredLogs = (() => {
    const range = getFilterRange(filter)
    return logs
      .filter((l) => {
        if (!range) return true
        // Untuk data dari Supabase, pakai tanggal string langsung
        const tStr = l.tanggal // format: "2026-04-26"
        const tDate = new Date(tStr + 'T00:00:00')
        return tDate >= range.start && tDate < range.end
      })
      .filter((l) => {
        if (!searchQuery) return true
        const q = searchQuery.toLowerCase()
        return l.nama.toLowerCase().includes(q) || l.nim.toLowerCase().includes(q)
      })
  })()

  const filterLabels = {
    hari_ini: 'Hari Ini',
    kemarin: 'Kemarin',
    '2_hari_lalu': '2 Hari Lalu',
    '1_bulan': '1 Bulan Terakhir',
  }

  const handleExport = async () => {
    if (filteredLogs.length === 0) return
    setExporting(true)
    try { await exportToExcel(filteredLogs, filterLabels[filter]) }
    finally { setExporting(false) }
  }

  // ─── Blokir: hapus user dari users_parkir di server ─────────
const handleBlokir = async (user) => {
  setBlokirAction(user.nim)
  const ok = await blockUser(user.nim)
  if (ok) {
    setRegisteredUsers((prev) =>
      prev.map((u) =>
        u.nim === user.nim ? { ...u, is_blocked: true } : u
      )
    )
  }
  setBlokirAction(null)
}

const handleUnblock = async (user) => {
  setBlokirAction(user.nim)
  const ok = await unblockUser(user.nim)
  if (ok) {
    setRegisteredUsers((prev) =>
      prev.map((u) =>
        u.nim === user.nim ? { ...u, is_blocked: false } : u
      )
    )
  }
  setBlokirAction(null)
}

  // ─── Filter Blokir User ──────────────────────────────────────
  const filteredBlokirUsers = registeredUsers.filter((u) => {
    if (!blokirSearchQuery) return true
    const q = blokirSearchQuery.toLowerCase()
    return u.nama.toLowerCase().includes(q) || u.nim.toLowerCase().includes(q)
  })

  // ─── RENDER ──────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-950 text-white" style={{ fontFamily: "'DM Sans', 'Segoe UI', sans-serif" }}>

      {/* NAVBAR */}
      <nav className="bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <img src={logo} alt="logo" className="w-8 h-8 object-contain" />
          <div>
            <p className="text-white font-bold text-sm leading-tight">Face Recognition Gate</p>
            <p className="text-gray-500 text-[11px]">Telkom University — Dashboard</p>
          </div>
        </div>
        <button   onClick={() => {
              localStorage.removeItem('isLoggedIn')
              navigate('/')
            }} 
            className="text-gray-400 hover:text-red-400 text-sm transition-all"
          >
            Logout →
        </button>
      </nav>

      {/* TAB MENU */}
      <div className="bg-gray-900/80 border-b border-gray-800 px-6 flex justify-center gap-2">
        {[
          { id: 'log', label: '📋 Log Akses' },
          { id: 'blokir', label: '🚫 Blokir User' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-5 py-3.5 text-sm font-medium transition-all duration-200 border-b-2 ${
              activeTab === tab.id
                ? 'border-red-500 text-red-400'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
        <button
          onClick={() => navigate('/registrasi')}
          className="px-5 py-3.5 text-sm font-medium transition-all duration-200 border-b-2 border-transparent text-gray-400 hover:text-white"
        >
          ➕ Registrasi User
        </button>
      </div>

      {/* KONTEN */}
      <div className="p-6 max-w-7xl mx-auto">

        {/* ══════ TAB: LOG AKSES ══════ */}
        {activeTab === 'log' && (
          <div className="flex flex-col gap-5">

            {/* Stat cards */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 text-center">
                <p className="text-3xl font-bold text-green-400">{stats.granted}</p>
                <p className="text-gray-500 text-xs mt-1 uppercase tracking-wider">Granted</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 text-center">
                <p className="text-3xl font-bold text-red-400">{stats.denied}</p>
                <p className="text-gray-500 text-xs mt-1 uppercase tracking-wider">Denied</p>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4 text-center">
                <p className="text-3xl font-bold text-blue-400">{stats.total}</p>
                <p className="text-gray-500 text-xs mt-1 uppercase tracking-wider">Total</p>
              </div>
            </div>

            {/* Status + Display Gate */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3 bg-gray-800/70 border border-gray-700 rounded-xl px-4 py-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className={`text-xs font-mono ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                  {wsStatus}
                </span>
              </div>
              <button
                onClick={() => window.open('/gate', '_blank')}
                className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-gray-200 hover:text-white px-4 py-2 rounded-xl text-sm font-medium transition-all"
              >
                📺 Display Gate ↗
              </button>
            </div>

            {/* Filter + Search + Export */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
              <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(filterLabels).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => setFilter(key)}
                      className={`text-xs px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                        filter === key
                          ? 'bg-red-600 text-white shadow-lg shadow-red-900/40'
                          : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 border border-gray-700'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2 w-full sm:w-auto">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Cari nama / NIM..."
                    className="flex-1 sm:w-52 bg-gray-800 border border-gray-700 text-white text-sm rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600"
                  />
                  <button
                    onClick={handleExport}
                    disabled={filteredLogs.length === 0 || exporting}
                    className={`flex items-center gap-2 text-xs font-semibold px-4 py-2 rounded-lg transition-all ${
                      filteredLogs.length === 0 || exporting
                        ? 'bg-gray-800 text-gray-600 border border-gray-700 cursor-not-allowed'
                        : 'bg-emerald-700 hover:bg-emerald-600 text-white'
                    }`}
                  >
                    {exporting
                      ? <><span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />Exporting...</>
                      : <>📊 Export Excel</>}
                  </button>
                </div>
              </div>
              <p className="text-gray-600 text-xs mt-3">
                Menampilkan <span className="text-gray-400 font-semibold">{filteredLogs.length}</span> entri
                {searchQuery && <> untuk "<span className="text-gray-400">{searchQuery}</span>"</>}
                {' '}— <span className="text-gray-400">{filterLabels[filter]}</span>
              </p>
            </div>

            {/* Tabel Log */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-800 flex items-center justify-between">
                <div>
                  <p className="font-semibold text-white text-sm">Log Aktivitas Akses</p>
                  <p className="text-gray-500 text-xs mt-0.5">
                    {loadingLogs ? 'Memuat...' : 'Tersimpan permanen di cloud ☁️'}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
                  <span className="text-xs text-gray-500">{isConnected ? 'Live' : 'Offline'}</span>
                </div>
              </div>

              {loadingLogs ? (
                <div className="py-20 text-center">
                  <span className="w-8 h-8 border-2 border-gray-700 border-t-red-500 rounded-full animate-spin inline-block mb-3" />
                  <p className="text-gray-500 text-sm">Memuat ...</p>
                </div>
              ) : filteredLogs.length === 0 ? (
                <div className="py-20 text-center">
                  <p className="text-4xl mb-3">📋</p>
                  <p className="text-gray-500 text-sm">
                    {logs.length === 0 ? 'Belum ada data.' : 'Tidak ada data untuk filter ini'}
                  </p>
                  {!isConnected && <p className="text-red-400 text-xs mt-2">⚠️ WebSocket tidak terhubung</p>}
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-800 text-gray-500 text-[11px] uppercase tracking-wider">
                        <th className="px-5 py-3 text-left">Nama</th>
                        <th className="px-5 py-3 text-left">NIM</th>
                        <th className="px-5 py-3 text-left">Waktu</th>
                        <th className="px-5 py-3 text-left hidden sm:table-cell">Tanggal</th>
                        <th className="px-5 py-3 text-left hidden lg:table-cell">Gate</th>
                        <th className="px-5 py-3 text-left hidden lg:table-cell">Confidence</th>
                        <th className="px-5 py-3 text-left">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredLogs.map((log) => (
                        <tr key={log.id} className="border-b border-gray-800/60 transition-colors hover:bg-gray-800/40">
                          <td className="px-5 py-3.5 text-sm text-white font-medium">{log.nama}</td>
                          <td className="px-5 py-3.5 text-sm text-gray-400 font-mono">{log.nim}</td>
                          <td className="px-5 py-3.5 text-sm text-gray-400 font-mono">{log.waktu}</td>
                          <td className="px-5 py-3.5 text-sm text-gray-500 hidden sm:table-cell">{log.tanggal}</td>
                          <td className="px-5 py-3.5 text-sm text-gray-500 hidden lg:table-cell">{log.gate}</td>
                          <td className="px-5 py-3.5 text-sm hidden lg:table-cell">
                            {log.status === 'GRANTED' && log.confidence ? (
                              <span className="text-gray-400 font-mono">
                                {(parseFloat(log.confidence) * 100).toFixed(1)}%
                              </span>
                            ) : <span className="text-gray-700">—</span>}
                          </td>
                          <td className="px-5 py-3.5"><StatusBadge status={log.status} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══════ TAB: BLOKIR USER ══════ */}
        {activeTab === 'blokir' && (
          <div className="flex flex-col gap-5">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-800">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div>
                    <p className="font-semibold text-white text-sm">Manajemen Blokir Pengguna</p>
                    <p className="text-gray-500 text-xs mt-0.5">
                      Data blokir tersimpan.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 w-full sm:w-auto">
                    <div className="relative flex-1 sm:w-56">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">🔍</span>
                      <input
                        type="text"
                        value={blokirSearchQuery}
                        onChange={(e) => setBlokirSearchQuery(e.target.value)}
                        placeholder="Cari nama / NIM..."
                        className="w-full bg-gray-800 border border-gray-700 text-white text-sm rounded-lg pl-8 pr-3 py-2 outline-none focus:ring-2 focus:ring-red-600/50 placeholder-gray-600"
                      />
                    </div>
                    {blokirSearchQuery && (
                      <button
                        onClick={() => setBlokirSearchQuery('')}
                        className="text-gray-500 hover:text-white text-xs px-2 py-2 rounded-lg bg-gray-800 border border-gray-700 transition-all"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>
                {!loadingBlokir && registeredUsers.length > 0 && (
                  <p className="text-gray-600 text-xs mt-2.5">
                    Menampilkan <span className="text-gray-400 font-semibold">{filteredBlokirUsers.length}</span> dari <span className="text-gray-400 font-semibold">{registeredUsers.length}</span> pengguna
                    {blokirSearchQuery && <> untuk "<span className="text-gray-400">{blokirSearchQuery}</span>"</>}
                  </p>
                )}
              </div>

              {loadingBlokir ? (
                <div className="py-20 text-center">
                  <span className="w-8 h-8 border-2 border-gray-700 border-t-red-500 rounded-full animate-spin inline-block mb-3" />
                  <p className="text-gray-500 text-sm">Memuat data blokir...</p>
                </div>
              ) : registeredUsers.length === 0 ? (
                <div className="py-20 text-center">
                  <p className="text-4xl mb-3">👤</p>
                  <p className="text-gray-500 text-sm">Belum ada pengguna terdaftar</p>
                </div>
              ) : filteredBlokirUsers.length === 0 ? (
                <div className="py-20 text-center">
                  <p className="text-4xl mb-3">🔍</p>
                  <p className="text-gray-500 text-sm">Tidak ada pengguna dengan kata kunci "<span className="text-gray-400">{blokirSearchQuery}</span>"</p>
                  <button onClick={() => setBlokirSearchQuery('')} className="mt-3 text-xs text-red-400 hover:text-red-300 underline">
                    Hapus filter pencarian
                  </button>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-800 text-gray-500 text-[11px] uppercase tracking-wider">
                        <th className="px-5 py-3 text-left">Nama</th>
                        <th className="px-5 py-3 text-left">NIM</th>
                        <th className="px-5 py-3 text-left hidden md:table-cell">Jam Terakhir</th>
                        <th className="px-5 py-3 text-left hidden sm:table-cell">Tanggal</th>
                        <th className="px-5 py-3 text-left">Status</th>
                        <th className="px-5 py-3 text-left">Aksi</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredBlokirUsers.map((user) => {
                        const isProcessing = blokirAction === user.nim
                        return (
                          <tr key={user.nim} className="border-b border-gray-800/60 hover:bg-gray-800/40 transition-colors">
                            <td className="px-5 py-3.5 text-sm text-white font-medium">{user.nama}</td>
                            <td className="px-5 py-3.5 text-sm text-gray-400 font-mono">{user.nim}</td>
                            <td className="px-5 py-3.5 text-sm text-gray-500 hidden md:table-cell">{user.jam_terakhir || '—'}</td>
                            <td className="px-5 py-3.5 text-sm text-gray-500 hidden sm:table-cell">{user.tanggal_terakhir || '—'}</td>
                            <td className="px-5 py-3.5">
                              {user.is_blocked ? (
                                <span className="text-xs font-bold px-3 py-1 rounded-full bg-red-900/60 text-red-400 border border-red-500/30">BLOKIR</span>
                              ) : (
                                <span className="text-xs font-bold px-3 py-1 rounded-full bg-green-900/40 text-green-500 border border-green-500/30">AKTIF</span>
                              )}
                            </td>
                            <td className="px-5 py-3.5">
                              {user.is_blocked ? (
                                <button
                                  onClick={() => handleUnblock(user)}
                                  disabled={isProcessing}
                                  className="flex items-center gap-1.5 text-xs bg-green-900/60 hover:bg-green-800 text-green-400 hover:text-white px-3 py-1.5 rounded-lg border border-green-500/40 transition-all disabled:opacity-50"
                                >
                                  {isProcessing ? <span className="w-3 h-3 border border-green-400/30 border-t-green-400 rounded-full animate-spin" /> : '🔓'} Unblock
                                </button>
                              ) : (
                                <button
                                  onClick={() => handleBlokir(user)}
                                  disabled={isProcessing}
                                  className="flex items-center gap-1.5 text-xs bg-red-900/60 hover:bg-red-800 text-red-400 hover:text-white px-3 py-1.5 rounded-lg border border-red-500/40 transition-all disabled:opacity-50"
                                >
                                  {isProcessing ? <span className="w-3 h-3 border border-red-400/30 border-t-red-400 rounded-full animate-spin" /> : '🚫'} Blokir
                                </button>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPage