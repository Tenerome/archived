<%@ page contentType="text/html; charset=UTF-8" %>
<%@ page import="entity.student" %>
<%@ page import="entity.db" %>
<%
String id=request.getParameter("id")==null?"":request.getParameter("id");
student stu = db.getStudentByID(id);
%>
<meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="./css/main.css">
<script src="./js/content.js"></script>
<script src="./js/particles.js" ></script>
<title>
修改
</title>
<div id="particles-js">
<canvas class="particles-js-canvas-el"></canvas>
<div class="frame">
	<br><br>
	<h2>UPDATE</h2>
<div id="enter">
	<form  name='enterstudent' action="update" method="post">
	        <input type="hidden" name="id" value=<%=stu.getId() %>>
		<div>
		<input type="text" id="sno" name="sno" value=<%=stu.getSno() %>>
		</div>
		<div>
			<input type="text" id="sname" name="sname" value=<%=stu.getSname() %>>
		</div>
		
		<div>
			<input type="text" id="smajor" name="smajor" value=<%=stu.getSmajor() %>>			    
		</div>
		<div>
			<input type="text" id="sgrade" name="sgrade" maxlength="4" value=<%=stu.getSgrade() %>>	
		</div>	
		<div id="selector">
			<input type="radio" id="ssex" name="ssex" value="男" <%=stu.getSsex().equals("男")? "checked":"" %> style="margin-left:20px;"> 
			<label>男</label>

			<input type="radio" id="ssex" name="ssex" value="女" <%=stu.getSsex().equals("女")?"checked":"" %> style="margin-left:200px;"><label>女</label>
		</div>
		<div class="addbt" style="border:0;">
			<input type="submit" value="修改" />
			<input type="reset" value="取消" />
		</div>
		
	</form>
</div>
</div>
</div> 