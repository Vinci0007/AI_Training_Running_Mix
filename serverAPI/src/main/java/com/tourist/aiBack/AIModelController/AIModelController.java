package com.tourist.aiBack.AIModelController;

// import javax.servlet.http.HttpSession;
import javax.validation.Valid;
// import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.MessageSource;

import com.tourist.aiBack.AIModelController.bean.ModelConfig;

@RestController
@RequestMapping("/v1/userAccount")
public class AIModelController {


    private static final Logger logger = LoggerFactory.getLogger(ModelConfig.class);

    @Autowired(required = true)
    MessageSource messageSource;

    // @Autowired(required = true)
    // AdminAccountService adminAccountService;

    // @Autowired(required = true)
    // AdminPermissionService adminPermissionService;

    // @Autowired(required = true)
    // AdminAccessLogService adminAccessLogService;

    // @Autowired(required = true)
    // HttpSession session;
}
