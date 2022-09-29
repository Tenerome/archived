<%@ page contentType="text/html; charset=UTF-8" %>
<%
	String login=(String)session.getAttribute("login");
%>
<html>
	<head>
		<meta charset="utf-8" />
		<title>登录</title>
		<link rel="shortcut icon" href="./pic/icon/door.ico">
		<link rel="stylesheet" type="text/css"href="./css/index.css"/>
		<link rel="stylesheet" type="text/css"href="./css/iconfont.css"/>
	</head>
	<body>
		<div id="login-box">
			<h1>Login</h1>
			<form id='login' name='login' action="login" method="post">
			<div class="input-box">
				<i class="iconfont">&#xe609;</i>
				
				<input type="text" placeholder="UserName" name="username"/>
			</div>
			<div class="input-box">
				<i class="iconfont">&#xe605;</i>
				<input type="password" placeholder="UserPassword" name="password" />
			</div>
			<button type="button" onclick="checknull()">登录</button>
			<br>
			<br>
			<a href="register.jsp">还没有账号?</a>
			</form>
		</div>
		<div id="foot">  <%-- foot--%>
			2021©Tenerome&nbsp;&nbsp;&nbsp;Email:ch.heng@QQ
		</div>
		
	</body>
<script type = "text/javascript">
        window.onload = function()
        {
            var ss="<%=login %>";
			if(ss=="false")
				alert("账号或密码错误");
        }
		function checknull() {
      var name =document.login.username;
      var pass=document.login.password;
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

