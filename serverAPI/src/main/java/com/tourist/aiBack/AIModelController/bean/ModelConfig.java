package com.tourist.aiBack.AIModelController.bean;

public class ModelConfig {

    private String modelName;
    private String modelPath;
    private String modelType;
    private String modelDesc;

    private boolean isLocalRunning;
    private String remoteRunningUrl;
    private String remoteRunningPort;

    public String getModelName() {
        return modelName;
    }
    public String getModelPath() {
        return modelPath;
    }
    public String getModelType() {
        return modelType;
    }
    public String getModelDesc() {
        return modelDesc;
    }
    public boolean isLocalRunning() {
        return isLocalRunning;
    }
    public String getRemoteRunningUrl() {
        return remoteRunningUrl;
    }
    public String getRemoteRunningPort() {
        return remoteRunningPort;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }
    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }
    public void setModelType(String modelType) {
        this.modelType = modelType;
    }
    public void setModelDesc(String modelDesc) {
        this.modelDesc = modelDesc;
    }
    public void setLocalRunning(boolean isLocalRunning) {
        this.isLocalRunning = isLocalRunning;
    }
    public void setRemoteRunningUrl(String remoteRunningUrl) {
        this.remoteRunningUrl = remoteRunningUrl;
    }
    public void setRemoteRunningPort(String remoteRunningPort) {
        this.remoteRunningPort = remoteRunningPort;
    }



}
