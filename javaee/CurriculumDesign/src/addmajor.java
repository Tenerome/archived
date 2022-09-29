package	servlets;
import entity.*;
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.*;

@WebServlet("/addmajor")
public class addmajor extends HttpServlet {
    public addmajor(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");
        String info="";
		if(new major(request).add()) {
			info="添加成功";
		}else {
			info="添加失败";
		}
		request.setAttribute("info", info);
		String url="index.jsp?operate=output.jsp";
		request.getRequestDispatcher(url).forward(request, response);
}
}
