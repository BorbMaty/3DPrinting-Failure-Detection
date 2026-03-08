importScripts("https://www.gstatic.com/firebasejs/11.4.0/firebase-app-compat.js");
importScripts("https://www.gstatic.com/firebasejs/11.4.0/firebase-messaging-compat.js");

firebase.initializeApp({
  apiKey:            "api-key",
  authDomain:        "printermonitor-488112.firebaseapp.com",
  projectId:         "printermonitor-488112",
  storageBucket:     "printermonitor-488112.firebasestorage.app",
  messagingSenderId: "895714392909",
  appId:             "1:895714392909:web:553314befa3149f6ad1504",
});

const messaging = firebase.messaging();

// Handle background messages (when tab is not focused)
messaging.onBackgroundMessage((payload) => {
  const { title, body, icon } = payload.notification ?? {};
  self.registration.showNotification(title ?? "PrinterMonitor Alert", {
    body:  body  ?? "A defect was detected.",
    icon:  icon  ?? "/favicon.ico",
    badge: "/favicon.ico",
    tag:   "printermonitor-alert",  // replaces previous notification
    renotify: true,
    data: payload.data ?? {},
  });
});
