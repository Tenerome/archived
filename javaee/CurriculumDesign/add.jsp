<%@ page contentType="text/html; charset=UTF-8" %>
<%@ page import="entity.major" %>
<% major major=new major(); %>
<meta charset="UTF-8">
<link rel="stylesheet" type="text/css" href="./css/main.css">
<script src="./js/content.js"></script>
<script src="./js/particles.js" ></script>
<title>
添加
</title>
<div id="particles-js">
<canvas class="particles-js-canvas-el"></canvas>
<div class="frame">
	<br><br>
	<h2>ADD</h2>
<div id="enter">
	<form name='enterstudent' action="add" method="post">
		<div>
			<input type="text" id="sno" name="sno" placeholder="学号">
		</div>
		<div>
			<input type="text" id="sname" name="sname" placeholder="姓名">
		</div>

		<div>
			<select id="smajor" name="smajor">
			     <option value="-1">选择专业</option>
			     <%=major.getMajorforSelect()%>   
			</select>			
		</div>
		<div>
			<input type="text" id="sgrade" name="sgrade" maxlength="4" placeholder="成绩">	
		</div>		
		<div id="selector">
			<input type="radio" id="ssex" name="ssex" value="男"style="margin-left:20px;"><label>男</label>
			<input type="radio" id="ssex" name="ssex" value="女"style="margin-left:200px;"><label>女</label>
		</div>
		<div class="addbt" style="border:0;">
			<input type="submit" value="添加" />
			<input type="reset" value="取消" />
		</div>
	</form>
</div>
</div>
</div> 