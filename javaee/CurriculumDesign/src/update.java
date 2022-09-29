package	servlets;
import entity.*;

import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;

@WebServlet("/update")
public class update extends HttpServlet {
    public update(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");
        String info="";
		if(db.update(new student(request))) {
			info="修改成功";
		}else {
			info="修改失败";
		}
		request.setAttribute("info", info);
		String url="index.jsp?operate=output.jsp";
		request.getRequestDispatcher(url).forward(request, response);
}
}
