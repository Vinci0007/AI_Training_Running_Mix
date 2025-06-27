// 登录页面
import React, { useState } from "react";
// import { DesktopOutlined, LockOutlined } from "@ant-design/icons";

import connectServer from "./utils/remote/serverConnect";

const HomeIndex = () => {
  const [isLocalRunning, setIsLocalRunning] = useState(true);
  const [remoteAddress, setRemoteAddress] = useState("");
  const [remotePort, setRemotePort] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  // 通过页面组件选择是本地运行还是远程运行
  const handleDeviceSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setIsLocalRunning(value === "local");
  };

  // 本地运行
  const handleLocalRun = () => {
    // 本地运行
    // 这里应该调用后端接口启动本地运行
    setIsLocalRunning(true);
  };

  // 远程运行
  const handleRemoteRun = () => {
    // 远程运行
    connectServer(remoteAddress, remotePort, username, password)
    // 这里应该调用后端接口启动远程运行
    setIsLocalRunning(false);
  };

  // 远程地址输入框

  const getDeviceStatus = () => {
    // 获取设备状态
    // 这里应该调用后端接口获取设备状态
    setIsLocalRunning(true);
  };
}