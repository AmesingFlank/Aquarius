#include "ui.h"

#include <memory>

#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_STYLE

#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT


#include <nuklear.h>

#include <nuklear_glfw_gl3.h>

#include "../Common/GpuCommons.h"

#include "../Rendering/WindowInfo.h"



void set_style(struct nk_context* ctx)
{
		nk_color table[NK_COLOR_COUNT];
	
		nk_color themeColor = nk_rgba(50, 50,50, 255);
		nk_color buttonColor = nk_rgba(48, 83, 111, 255);

		table[NK_COLOR_TEXT] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_WINDOW] = nk_rgba(60,60,60, 255);
		table[NK_COLOR_HEADER] = nk_rgba(51, 51, 56, 220);
		table[NK_COLOR_BORDER] = nk_rgba(46, 46, 46, 255);
		table[NK_COLOR_BUTTON] =buttonColor;
		table[NK_COLOR_BUTTON_HOVER] = nk_rgba(58, 93, 121, 255);
		table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(63, 98, 126, 255);
		table[NK_COLOR_TOGGLE] = themeColor;
		table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 53, 56, 255);
		table[NK_COLOR_TOGGLE_CURSOR] =buttonColor;
		table[NK_COLOR_SELECT] = nk_rgba(57, 67, 61, 255);
		table[NK_COLOR_SELECT_ACTIVE] =buttonColor;
		table[NK_COLOR_SLIDER] = themeColor;
		table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(48, 83, 111, 245);
		table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
		table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
		table[NK_COLOR_PROPERTY] = themeColor;
		table[NK_COLOR_EDIT] = nk_rgba(50, 58, 61, 225);
		table[NK_COLOR_EDIT_CURSOR] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_COMBO] = themeColor;
		table[NK_COLOR_CHART] = themeColor;
		table[NK_COLOR_CHART_COLOR] =buttonColor;
		table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
		table[NK_COLOR_SCROLLBAR] = themeColor;
		table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(200,200,200, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(200, 200, 200, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(200, 200, 200, 255);
		table[NK_COLOR_TAB_HEADER] =buttonColor;
		nk_style_from_table(ctx, table);
	
}

nk_context* createUI(GLFWwindow* win) {
	WindowInfo& windowInfo = WindowInfo::instance();
	int windowHeight = windowInfo.windowHeight;
	int windowWidth = windowInfo.windowWidth;

	struct nk_context* ctx;

	ctx = nk_glfw3_init(win, NK_GLFW3_INSTALL_CALLBACKS);
	/* Load Fonts: if none of these are loaded a default font will be used  */
	/* Load Cursor: if you uncomment cursor loading please hide the cursor */
	struct nk_font_atlas* atlas;
	
	nk_glfw3_font_stash_begin(&atlas);
	nk_font* calibri = nk_font_atlas_add_from_file(atlas, "./resources/Fonts/opensans.ttf", windowHeight*0.025, 0);

	nk_glfw3_font_stash_end();
	//nk_style_load_all_cursors(ctx, atlas->cursors);
	nk_style_set_font(ctx, &calibri->handle);

	

	set_style(ctx);

	return ctx;
	
}

void drawUI(nk_context* ctx, FluidConfig& fluidConfig,std::function<void()> onStart) 
{
	WindowInfo& windowInfo = WindowInfo::instance();
	int windowHeight = windowInfo.windowHeight;
	int windowWidth = windowInfo.windowWidth;

	nk_glfw3_new_frame();

	

	float incPerPixel = 200.f;

	float widgetBoundary = windowWidth * 0.02;
	float widgetWidth = windowWidth * 0.2;

	


	float rightSideWidgetBegin = windowWidth - widgetWidth - widgetBoundary;

	float rowHeight = windowHeight * 0.02;

#define GAP nk_layout_row_dynamic(ctx, rowHeight, 1);nk_label(ctx, "", NK_TEXT_LEFT)
#define GAP_SMALL nk_layout_row_dynamic(ctx, rowHeight*0.5, 1);nk_label(ctx, "", NK_TEXT_LEFT)

	if (nk_begin(ctx, "Simulation Set-up", nk_rect(widgetBoundary, widgetBoundary, widgetWidth, windowHeight*0.9),
		NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
		NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
	{
		nk_layout_row_dynamic(ctx, rowHeight, 1);
		if (nk_button_label(ctx, "Start / Restart Simulation")) {
			onStart();
		}

		GAP_SMALL;
		

		nk_layout_row_dynamic(ctx, rowHeight, 1);
		nk_label(ctx, "Simulation Algorithm:", NK_TEXT_LEFT);

		nk_layout_row_dynamic(ctx, rowHeight, 3);
		if (nk_option_label(ctx, "FLIP", fluidConfig.method=="FLIP" ))fluidConfig.method = "FLIP";
		if (nk_option_label(ctx, "PBF", fluidConfig.method == "PBF"))fluidConfig.method = "PBF";
		if (nk_option_label(ctx, "PCISPH", fluidConfig.method == "PCISPH"))fluidConfig.method = "PCISPH";

		GAP;


		bool isFLIP = fluidConfig.method == "FLIP";



		nk_layout_row_dynamic(ctx, rowHeight, 1);
		nk_label(ctx, "Gravity:", NK_TEXT_LEFT);
		nk_layout_row_dynamic(ctx, rowHeight, 3);

		nk_property_float(ctx, "g:", -10, &fluidConfig.gravity.x, 10, 0.2, incPerPixel);
		nk_property_float(ctx, "g:", -10, &fluidConfig.gravity.y, 10, 0.2, incPerPixel);
		nk_property_float(ctx, "g:", -10, &fluidConfig.gravity.z, 10, 0.2, incPerPixel);
		GAP;

		

		if (nk_tree_push(ctx, NK_TREE_NODE, "Initial Fluid Volumes:", NK_MAXIMIZED))
		{
			nk_layout_row_dynamic(ctx, rowHeight, 1);
			if (nk_button_label(ctx, "Add New Initial Volume")) {
				fluidConfig.initialVolumes.emplace_back();
			}
			GAP_SMALL;
			std::vector<InitializationVolume>::iterator it = fluidConfig.initialVolumes.begin();
			int id = 0;
			for (; it != fluidConfig.initialVolumes.end();) {
				InitializationVolume& volume = *it;
				bool erased = false;
				if (nk_tree_push(ctx, NK_TREE_NODE, std::to_string(id).c_str(), NK_MAXIMIZED))
				{
					nk_layout_row_dynamic(ctx, rowHeight, 3);
					nk_label(ctx, "Type:", NK_TEXT_LEFT);
					

					float incStep = 1.f/16;
					


					if (nk_option_label(ctx, "Box", volume.shapeType == ShapeType::Square)) {
						volume.shapeType = ShapeType::Square;
						while (volume.params.size() < 6) {
							volume.params.push_back(0);
						}
					}
						
					if (nk_option_label(ctx, "Ball", volume.shapeType == ShapeType::Sphere)) {
						volume.shapeType = ShapeType::Sphere;
						while (volume.params.size() < 4) {
							volume.params.push_back(0);
						}
					}
					GAP_SMALL;

					if (volume.shapeType == ShapeType::Square) {
						nk_layout_row_dynamic(ctx, rowHeight, 1);
						nk_label(ctx, "Min Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, rowHeight, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[0], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[1], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[2], 1, incStep, incPerPixel);
						GAP_SMALL;

						nk_layout_row_dynamic(ctx, rowHeight, 1);
						nk_label(ctx, "Max Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, rowHeight, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[3], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[4], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[5], 1, incStep, incPerPixel);
						GAP_SMALL;

						if (isFLIP) {
							nk_layout_row_dynamic(ctx, rowHeight, 3);
							nk_label(ctx, "Phase:", NK_TEXT_LEFT);
							nk_property_int(ctx, "", 0, &volume.phase, fluidConfig.phaseCount, 1, incPerPixel);
							GAP_SMALL;
						}
						nk_layout_row_dynamic(ctx, rowHeight, 1);
						if (nk_button_label(ctx, "delete")) {
							fluidConfig.initialVolumes.erase(it);
							erased = true;
						}
						
						GAP_SMALL;
					}
					
					else if (volume.shapeType == ShapeType::Sphere) {
						nk_layout_row_dynamic(ctx, rowHeight, 1);
						nk_label(ctx, "Center Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, rowHeight, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[0], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[1], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[2], 1, incStep, incPerPixel);
						GAP_SMALL;

						nk_layout_row_dynamic(ctx, rowHeight, 2);
						nk_label(ctx, "Radius:", NK_TEXT_LEFT);
						nk_property_float(ctx, "r:", 0, &volume.params[3], 1, incStep, incPerPixel);
						GAP_SMALL;
						if (isFLIP) {
							nk_layout_row_dynamic(ctx, rowHeight, 3);
							nk_label(ctx, "Phase:", NK_TEXT_LEFT);
							nk_property_int(ctx, "", 0, &volume.phase, fluidConfig.phaseCount, 1, incPerPixel);
							GAP_SMALL;
						}
						nk_layout_row_dynamic(ctx, rowHeight, 1);
						if (nk_button_label(ctx, "delete")) {
							fluidConfig.initialVolumes.erase(it);
							erased = true;
						}
						GAP_SMALL;
					}

					nk_tree_pop(ctx);
					
				}
				if (!erased) {
					++it; 
					++id;
				}
				
			}
			

			nk_tree_pop(ctx);
		}
		

	}
	nk_end(ctx);


	float algoWidgetTop = widgetBoundary;
	float algoWidgetHeight = windowHeight * 0.28;

	if (fluidConfig.method == "PCISPH") {
		algoWidgetHeight = windowHeight * 0.33;
	}


	float phaseWidgetTop = windowHeight * 0.35;
	float phaseWidgetHeight = windowHeight * 0.35;

	float instructionsWidgetTop = windowHeight * 0.73;
	float instructionsWidgetHeight = windowHeight * 0.25;


	

	if (fluidConfig.method == "FLIP") {
		if (nk_begin(ctx, "FLIP Settings", nk_rect(rightSideWidgetBegin, algoWidgetTop, widgetWidth, algoWidgetHeight),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
			NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {

			float timestep_ms = fluidConfig.FLIP.timestep * 1000;
			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Timestep(ms)", NK_TEXT_LEFT);
			nk_property_float(ctx, "dt:", 5, &timestep_ms, 50, 5, incPerPixel);
			fluidConfig.FLIP.timestep = timestep_ms / 1000;

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Grid Size: (for all of x/y/z)", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 30, &fluidConfig.FLIP.sizeX, 70, 5, incPerPixel);

			fluidConfig.FLIP.sizeY = fluidConfig.FLIP.sizeX;
			fluidConfig.FLIP.sizeZ = fluidConfig.FLIP.sizeX;

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Pressure Jacobian Solver Iterations:", NK_TEXT_ALIGN_LEFT);
			nk_property_int(ctx, "", 30, &fluidConfig.FLIP.pressureIterations, 200, 10, incPerPixel);

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Diffusion Jacobian Solver Iterations:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 30, &fluidConfig.FLIP.diffusionIterations, 200, 10, incPerPixel);
			
		}
	}
	else if (fluidConfig.method == "PBF") {
		if (nk_begin(ctx, "PBF Settings", nk_rect(rightSideWidgetBegin, algoWidgetTop, widgetWidth, algoWidgetHeight),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
			NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {

			float timestep_ms = fluidConfig.PBF.timestep * 1000;
			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Timestep(ms)", NK_TEXT_LEFT);
			nk_property_float(ctx, "dt:", 5, &timestep_ms, 50, 5, incPerPixel);
			fluidConfig.PBF.timestep = timestep_ms / 1000;

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Substeps:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 1, &fluidConfig.PBF.substeps, 10, 1, incPerPixel);

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Iterations:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 1, &fluidConfig.PBF.iterations, 10, 1, incPerPixel);

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			int count = fluidConfig.PBF.maxParticleCount / 1000;
			nk_label(ctx, "Max Particle Count (k)", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 100, &count, 1000, 100, incPerPixel);
			fluidConfig.PBF.maxParticleCount = count * 1000;
		}
	}

	else if (fluidConfig.method == "PCISPH") {
		if (nk_begin(ctx, "PCISPH Settings", nk_rect(rightSideWidgetBegin, algoWidgetTop, widgetWidth, algoWidgetHeight),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
			NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {

			float timestep_ms = fluidConfig.PCISPH.timestep * 1000;
			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Timestep(ms)", NK_TEXT_LEFT);
			nk_property_float(ctx, "dt:", 1, &timestep_ms, 20, 1, incPerPixel);
			fluidConfig.PCISPH.timestep = timestep_ms / 1000;

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Substeps:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 1, &fluidConfig.PCISPH.substeps, 10, 1, incPerPixel);

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Iterations:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 1, &fluidConfig.PCISPH.iterations, 10, 1, incPerPixel);

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			int count = fluidConfig.PCISPH.maxParticleCount / 1000;
			nk_label(ctx, "Max Particle Count (k)", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 100, &count, 1000, 100, incPerPixel);
			fluidConfig.PCISPH.maxParticleCount = count * 1000;

			GAP_SMALL;

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "stiffness:", NK_TEXT_LEFT);
			nk_property_float(ctx, "", 1, &fluidConfig.PCISPH.stiffness, 100, 5, incPerPixel);
		}
	}

	nk_end(ctx);

	if (fluidConfig.method == "FLIP") {
		if (nk_begin(ctx, "Multiphase Settings", nk_rect(rightSideWidgetBegin, phaseWidgetTop, widgetWidth, phaseWidgetHeight),
			NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
			NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE)) {

			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Phase Count:", NK_TEXT_LEFT);
			nk_property_int(ctx, "", 1, &fluidConfig.phaseCount, 4, 1, incPerPixel);

			GAP_SMALL;

			float log = log10(fluidConfig.diffusionCoeff);
			nk_layout_row_dynamic(ctx, rowHeight, 1);
			nk_label(ctx, "Diffusion Coeff (log10):", NK_TEXT_LEFT);
			nk_property_float(ctx, "", -4, &log, 1, 0.5, incPerPixel);
			fluidConfig.diffusionCoeff = pow(10, log);

			while (fluidConfig.phaseColors.size() < fluidConfig.phaseCount) {
				fluidConfig.phaseColors.push_back(make_float4(0, 0, 0, 0));
			}

			if (nk_tree_push(ctx, NK_TREE_NODE, "Phase Colors", NK_MAXIMIZED))
			{
				for (int i = 0; i < fluidConfig.phaseCount; ++i) {
					if (nk_tree_push(ctx, NK_TREE_NODE, std::to_string(i).c_str(), NK_MAXIMIZED)) {
						struct nk_colorf color;
						color.r = fluidConfig.phaseColors[i].x;
						color.g = fluidConfig.phaseColors[i].y;
						color.b = fluidConfig.phaseColors[i].z;
						color.a = fluidConfig.phaseColors[i].w;

						nk_layout_row_dynamic(ctx, rowHeight, 1);
						nk_label(ctx, "background:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, rowHeight, 1);
						if (nk_combo_begin_color(ctx, nk_rgb_cf(color), nk_vec2(nk_widget_width(ctx), windowHeight*0.3))) {
							nk_layout_row_dynamic(ctx, rowHeight*13, 1);
							color = nk_color_picker(ctx, color, NK_RGBA);
							nk_layout_row_dynamic(ctx, rowHeight, 1);
							color.r = nk_propertyf(ctx, "#R:", 0, color.r, 1.0f, 0.05f, 0.005f);
							color.g = nk_propertyf(ctx, "#G:", 0, color.g, 1.0f, 0.05f, 0.005f);
							color.b = nk_propertyf(ctx, "#B:", 0, color.b, 1.0f, 0.05f, 0.005f);
							color.a = nk_propertyf(ctx, "#A:", 0, color.a, 10.0f, 0.05f, 0.005f);
							nk_combo_end(ctx);
						}


						fluidConfig.phaseColors[i].x = color.r;
						fluidConfig.phaseColors[i].y = color.g;
						fluidConfig.phaseColors[i].z = color.b;
						fluidConfig.phaseColors[i].w = color.a;

						nk_tree_pop(ctx);
					}

				}
				nk_tree_pop(ctx);
			}
			
			nk_end(ctx);
		}
	}

	if (nk_begin(ctx, "Instructions", nk_rect(rightSideWidgetBegin, instructionsWidgetTop, widgetWidth, instructionsWidgetHeight),
		NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
		NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
	{
		

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_label(ctx, "SPACE to pause simulation", NK_TEXT_LEFT);

		GAP;

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_label(ctx, "SHIFT To switch render mode", NK_TEXT_LEFT);

		GAP;

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_label(ctx, "Up/Down/Left/Right to hover", NK_TEXT_LEFT);

		GAP;

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_label(ctx, "W/A/S/D to move", NK_TEXT_LEFT);

		GAP;

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_label(ctx, "Left mouse + drag to look around", NK_TEXT_LEFT);


	}
	nk_end(ctx);


#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
}