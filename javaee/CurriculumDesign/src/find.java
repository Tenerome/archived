package servlets;
import entity.*;

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;


@WebServlet("/find")

public class find extends HttpServlet {
    public find(){
        super();
    }
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        doPost(request, response);
    }

public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        request.setCharacterEncoding("UTF-8");
        response.setContentType("text/html;charset=UTF-8");
        db.getConnection();
        String info=db.getStuHTML(new student(request),0,0);
        request.setAttribute("info", info);
		String url="index.jsp?operate=output.jsp";
		request.getRequestDispatcher(url).forward(request, response);

}
}
