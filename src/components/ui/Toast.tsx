import { X, CheckCircle, AlertCircle, Info } from "lucide-react";
import styles from "./Toast.module.css";

interface ToastProps {
  msg: string;
  type: "success" | "error" | "info";
  onClose: () => void;
}

const icons = {
  success: CheckCircle,
  error: AlertCircle,
  info: Info,
};

export function Toast({ msg, type, onClose }: ToastProps) {
  const Icon = icons[type];
  return (
    <div className={`${styles.toast} ${styles[type]}`}>
      <Icon size={15} className={styles.icon} />
      <span className={styles.msg}>{msg}</span>
      <button className={styles.close} onClick={onClose}>
        <X size={13} />
      </button>
    </div>
  );
}
