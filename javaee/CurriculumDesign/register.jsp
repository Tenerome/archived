<%@ page language="java" contentType="text/html;chatset=UTF-8" pageEncoding="UTF-8"%>
<!doctype html>
<html>
<head>
<meta charset="UTF-8">
<link rel="shortcut icon" href="./pic/icon/door.ico">
<link rel="stylesheet" type="text/css"href="./css/register.css"/>
<link rel="stylesheet" type="text/css"href="./css/iconfont.css"/>
<title>
注册
</title>
</head>
<body>
<div id="login-box"> 
    <h1>Register</h1>
    <form action="register" method="post" name="form" id="login">
     	<div class="input-box">
				<i class="iconfont">&#xe609;</i>
				
				<input type="text" placeholder="UserName" name="username"/>
			</div>
			<div class="input-box">
				<i class="iconfont">&#xe605;</i>
				<input type="password" placeholder="UserPassword" name="password" />
			</div>   
    </form>
    <button type="button" id="regist" onclick="checknull()">注册</button>
</div>
</body>
<% 
String register=(String)request.getAttribute("register");
%>
<script>

    window.onload=function(){
        var regi="<%=register %>";
         if(regi=="false")
            alert("账号已存在");
    }
    
    function checknull() {
      var name =document.form.username;
      var pass=document.form.password;
      var subm=document.getElementById('login');
      if(name.value.length==0){
          alert("请输入账号");
      }else if(pass.value.length==0){
        alert("请输入密码");
        }else{
            subm.submit();
        }
   
    }
    
  
</script>
</html>