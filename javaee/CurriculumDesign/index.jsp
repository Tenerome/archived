<%@ page contentType="text/html; charset=UTF-8" %>
<html>
<head>
<title>选修课管理系统</title>
<link rel="shortcut icon" href="./pic/icon/book.ico">
<link rel="stylesheet" type="text/css" href="css/main.css" />
<link rel="stylesheet" type="text/css" href="css/table.css" />
</head>
<body>
    <%
    String login=(String)session.getAttribute("login");
    if(login==null)//第一次打开
        login="";
    if(!login.equals("true"))//排斥login为""和flase
        response.sendRedirect("login.jsp");
    %>
    <div id="top"><h1 class="header">ECMS</h1></div>
    <div>
    <div id="leftMenu">
        <h3>MENU</h3>
        <ul>
            <li><a href="index.jsp?operate=welcome.jsp"><img src="pic/icon/index.png">&nbsp;主页</a></li>
            <li><a href="index.jsp?operate=find.jsp"><img src="pic/icon/search.png">&nbsp;查询</a></li>
            <li><a href="index.jsp?operate=add.jsp"><img src="pic/icon/add.png">&nbsp;添加</a></li>
            <li><a href="index.jsp?operate=addmajor.jsp"><img src="pic/icon/set.png">&nbsp;专业维护</a></li>
            <li><a href="exit.jsp"><img src="pic/icon/exit.png">&nbsp;安全退出</a></li>
        </ul>
        <h3>FAVOURITE</h3>
        <ul>
            <li><a href="https://gitee.com/tenerome/code/tree/br1/javaee/CurriculumDesign" target="blank"><img src="pic/icon/code.png">&nbsp;Code</a></li>
            <li><a href="https://gitee.com/tenerome"  target="blank"><img src="pic/icon/follow.png">&nbsp;Fork me</a></li>
            <li><a href="mailto:ch.heng@qq.com" target="blank"><img src="pic/icon/email.png">&nbsp;ch.heng</a></li>
        </ul>
        <br><br><br><br><br><br><br><br>
        <br><br><br><br><br><br><br>
        <p style="color:#808080;font-size:7px;">&nbsp;Copyright&nbsp;©Tenerome</p>
    </div>
    <div id="content">  
        <jsp:include page='<%=request.getParameter("operate")==null?"welcome.jsp":request.getParameter("operate")%>'/>                
    </div>
    </div>
</body>
</html>