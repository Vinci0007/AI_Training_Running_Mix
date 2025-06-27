"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
// 登录页面
const react_1 = require("react");
const serverConnect_1 = __importDefault(require("./utils/remote/serverConnect"));
const HomeIndex = () => {
    const [isLocalRunning, setIsLocalRunning] = (0, react_1.useState)(true);
    const [remoteAddress, setRemoteAddress] = (0, react_1.useState)("");
    const [remotePort, setRemotePort] = (0, react_1.useState)("");
    const [username, setUsername] = (0, react_1.useState)("");
    const [password, setPassword] = (0, react_1.useState)("");
    // 通过页面组件选择是本地运行还是远程运行
    const handleDeviceSelect = (e) => {
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
        (0, serverConnect_1.default)(remoteAddress, remotePort, username, password);
        // 这里应该调用后端接口启动远程运行
        setIsLocalRunning(false);
    };
    // 远程地址输入框
    const getDeviceStatus = () => {
        // 获取设备状态
        // 这里应该调用后端接口获取设备状态
        setIsLocalRunning(true);
    };
};
