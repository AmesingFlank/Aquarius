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


enum theme { THEME_BLACK, THEME_WHITE, THEME_RED, THEME_BLUE, THEME_DARK };

void
set_style(struct nk_context* ctx, enum theme theme)
{
	struct nk_color table[NK_COLOR_COUNT];
	if (theme == THEME_WHITE) {
		table[NK_COLOR_TEXT] = nk_rgba(70, 70, 70, 255);
		table[NK_COLOR_WINDOW] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_HEADER] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_BORDER] = nk_rgba(0, 0, 0, 255);
		table[NK_COLOR_BUTTON] = nk_rgba(185, 185, 185, 255);
		table[NK_COLOR_BUTTON_HOVER] = nk_rgba(170, 170, 170, 255);
		table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(160, 160, 160, 255);
		table[NK_COLOR_TOGGLE] = nk_rgba(150, 150, 150, 255);
		table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(120, 120, 120, 255);
		table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_SELECT] = nk_rgba(190, 190, 190, 255);
		table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_SLIDER] = nk_rgba(190, 190, 190, 255);
		table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(80, 80, 80, 255);
		table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(70, 70, 70, 255);
		table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(60, 60, 60, 255);
		table[NK_COLOR_PROPERTY] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_EDIT] = nk_rgba(150, 150, 150, 255);
		table[NK_COLOR_EDIT_CURSOR] = nk_rgba(0, 0, 0, 255);
		table[NK_COLOR_COMBO] = nk_rgba(175, 175, 175, 255);
		table[NK_COLOR_CHART] = nk_rgba(160, 160, 160, 255);
		table[NK_COLOR_CHART_COLOR] = nk_rgba(45, 45, 45, 255);
		table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
		table[NK_COLOR_SCROLLBAR] = nk_rgba(180, 180, 180, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(140, 140, 140, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(150, 150, 150, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(160, 160, 160, 255);
		table[NK_COLOR_TAB_HEADER] = nk_rgba(180, 180, 180, 255);
		nk_style_from_table(ctx, table);
	}
	else if (theme == THEME_RED) {
		table[NK_COLOR_TEXT] = nk_rgba(190, 190, 190, 255);
		table[NK_COLOR_WINDOW] = nk_rgba(30, 33, 40, 215);
		table[NK_COLOR_HEADER] = nk_rgba(181, 45, 69, 220);
		table[NK_COLOR_BORDER] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_BUTTON] = nk_rgba(181, 45, 69, 255);
		table[NK_COLOR_BUTTON_HOVER] = nk_rgba(190, 50, 70, 255);
		table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(195, 55, 75, 255);
		table[NK_COLOR_TOGGLE] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 60, 60, 255);
		table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(181, 45, 69, 255);
		table[NK_COLOR_SELECT] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(181, 45, 69, 255);
		table[NK_COLOR_SLIDER] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(181, 45, 69, 255);
		table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(186, 50, 74, 255);
		table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(191, 55, 79, 255);
		table[NK_COLOR_PROPERTY] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_EDIT] = nk_rgba(51, 55, 67, 225);
		table[NK_COLOR_EDIT_CURSOR] = nk_rgba(190, 190, 190, 255);
		table[NK_COLOR_COMBO] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_CHART] = nk_rgba(51, 55, 67, 255);
		table[NK_COLOR_CHART_COLOR] = nk_rgba(170, 40, 60, 255);
		table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
		table[NK_COLOR_SCROLLBAR] = nk_rgba(30, 33, 40, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(64, 84, 95, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(70, 90, 100, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(75, 95, 105, 255);
		table[NK_COLOR_TAB_HEADER] = nk_rgba(181, 45, 69, 220);
		nk_style_from_table(ctx, table);
	}
	else if (theme == THEME_BLUE) {
		table[NK_COLOR_TEXT] = nk_rgba(20, 20, 20, 255);
		table[NK_COLOR_WINDOW] = nk_rgba(202, 212, 214, 215);
		table[NK_COLOR_HEADER] = nk_rgba(137, 182, 224, 220);
		table[NK_COLOR_BORDER] = nk_rgba(140, 159, 173, 255);
		table[NK_COLOR_BUTTON] = nk_rgba(137, 182, 224, 255);
		table[NK_COLOR_BUTTON_HOVER] = nk_rgba(142, 187, 229, 255);
		table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(147, 192, 234, 255);
		table[NK_COLOR_TOGGLE] = nk_rgba(177, 210, 210, 255);
		table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(182, 215, 215, 255);
		table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(137, 182, 224, 255);
		table[NK_COLOR_SELECT] = nk_rgba(177, 210, 210, 255);
		table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(137, 182, 224, 255);
		table[NK_COLOR_SLIDER] = nk_rgba(177, 210, 210, 255);
		table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(137, 182, 224, 245);
		table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(142, 188, 229, 255);
		table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(147, 193, 234, 255);
		table[NK_COLOR_PROPERTY] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_EDIT] = nk_rgba(210, 210, 210, 225);
		table[NK_COLOR_EDIT_CURSOR] = nk_rgba(20, 20, 20, 255);
		table[NK_COLOR_COMBO] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_CHART] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_CHART_COLOR] = nk_rgba(137, 182, 224, 255);
		table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
		table[NK_COLOR_SCROLLBAR] = nk_rgba(190, 200, 200, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(64, 84, 95, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(70, 90, 100, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(75, 95, 105, 255);
		table[NK_COLOR_TAB_HEADER] = nk_rgba(156, 193, 220, 255);
		nk_style_from_table(ctx, table);
	}
	else if (theme == THEME_DARK) {
		table[NK_COLOR_TEXT] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_WINDOW] = nk_rgba(57, 67, 71, 215);
		table[NK_COLOR_HEADER] = nk_rgba(51, 51, 56, 220);
		table[NK_COLOR_BORDER] = nk_rgba(46, 46, 46, 255);
		table[NK_COLOR_BUTTON] = nk_rgba(48, 83, 111, 255);
		table[NK_COLOR_BUTTON_HOVER] = nk_rgba(58, 93, 121, 255);
		table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(63, 98, 126, 255);
		table[NK_COLOR_TOGGLE] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(45, 53, 56, 255);
		table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(48, 83, 111, 255);
		table[NK_COLOR_SELECT] = nk_rgba(57, 67, 61, 255);
		table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(48, 83, 111, 255);
		table[NK_COLOR_SLIDER] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(48, 83, 111, 245);
		table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
		table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
		table[NK_COLOR_PROPERTY] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_EDIT] = nk_rgba(50, 58, 61, 225);
		table[NK_COLOR_EDIT_CURSOR] = nk_rgba(210, 210, 210, 255);
		table[NK_COLOR_COMBO] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_CHART] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_CHART_COLOR] = nk_rgba(48, 83, 111, 255);
		table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(255, 0, 0, 255);
		table[NK_COLOR_SCROLLBAR] = nk_rgba(50, 58, 61, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(48, 83, 111, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(53, 88, 116, 255);
		table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(58, 93, 121, 255);
		table[NK_COLOR_TAB_HEADER] = nk_rgba(48, 83, 111, 255);
		nk_style_from_table(ctx, table);
	}
	else {
		nk_style_default(ctx);
	}
}

nk_context* createUI(GLFWwindow* win) {
	WindowInfo& windowInfo = WindowInfo::instance();
	int windowHeight = windowInfo.windowHeight;
	int windowWidth = windowInfo.windowWidth;

	struct nk_context* ctx;
	struct nk_colorf bg;

	ctx = nk_glfw3_init(win, NK_GLFW3_INSTALL_CALLBACKS);
	/* Load Fonts: if none of these are loaded a default font will be used  */
	/* Load Cursor: if you uncomment cursor loading please hide the cursor */
	struct nk_font_atlas* atlas;
	
	nk_glfw3_font_stash_begin(&atlas);
	nk_font* calibri = nk_font_atlas_add_from_file(atlas, "./resources/Fonts/opensans.ttf", windowHeight*0.025, 0);

	/*struct nk_font *droid = nk_font_atlas_add_from_file(atlas, "../../../extra_font/DroidSans.ttf", 14, 0);*/
	/*struct nk_font *roboto = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Roboto-Regular.ttf", 14, 0);*/
	/*struct nk_font *future = nk_font_atlas_add_from_file(atlas, "../../../extra_font/kenvector_future_thin.ttf", 13, 0);*/
	// nk_font *clean = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyClean.ttf", 12, 0);
	/*struct nk_font *tiny = nk_font_atlas_add_from_file(atlas, "../../../extra_font/ProggyTiny.ttf", 10, 0);*/
	/*struct nk_font *cousine = nk_font_atlas_add_from_file(atlas, "../../../extra_font/Cousine-Regular.ttf", 13, 0);*/
	nk_glfw3_font_stash_end();
	nk_style_load_all_cursors(ctx, atlas->cursors);
	nk_style_set_font(ctx, &calibri->handle);


	//set_style(ctx, THEME_DARK);
#ifdef INCLUDE_STYLE
	/*set_style(ctx, THEME_WHITE);*/
	/*set_style(ctx, THEME_RED);*/
	/*set_style(ctx, THEME_BLUE);*/
	/*set_style(ctx, THEME_DARK);*/
#endif
	return ctx;
	
}

void drawUI(nk_context* ctx, FluidConfig& fluidConfig,std::function<void()> onStart) 
{
	WindowInfo& windowInfo = WindowInfo::instance();
	int windowHeight = windowInfo.windowHeight;
	int windowWidth = windowInfo.windowWidth;

	nk_glfw3_new_frame();

	struct nk_colorf bg;
	bg.r = 0.10f, bg.g = 0.18f, bg.b = 0.24f, bg.a = 1.0f;

	float incPerPixel = 200.f;

	if (nk_begin(ctx, "Simulation Set-up", nk_rect(windowWidth*0.02, windowWidth * 0.02, windowWidth * 0.2, windowHeight*0.9),
		NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
		NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
	{
		nk_layout_row_dynamic(ctx, 25, 1);
		if (nk_button_label(ctx, "Start / Restart Simulation")) {
			onStart();
		}

		
		

		nk_layout_row_dynamic(ctx, 40, 1);
		nk_label(ctx, "Simulation Algorithm:", NK_TEXT_LEFT);

		nk_layout_row_dynamic(ctx, 40, 3);
		if (nk_option_label(ctx, "FLIP", fluidConfig.method=="FLIP" ))fluidConfig.method = "FLIP";
		if (nk_option_label(ctx, "PBF", fluidConfig.method == "PBF"))fluidConfig.method = "PBF";
		if (nk_option_label(ctx, "PCISPH", fluidConfig.method == "PCISPH"))fluidConfig.method = "PCISPH";


		bool isFLIP = fluidConfig.method == "FLIP";

		nk_layout_row_dynamic(ctx, 25, 2);
		nk_label(ctx, "Gravity", NK_TEXT_LEFT);
		nk_property_float(ctx, "g:", -10, &fluidConfig.gravity, 10, 0.2, incPerPixel);

		nk_layout_row_dynamic(ctx, 25, 2);
		nk_label(ctx, "Timestep", NK_TEXT_LEFT);
		nk_property_float(ctx, "dt:", 0.005, &fluidConfig.timestep, 0.05, 0.005, incPerPixel);


		if (nk_tree_push(ctx, NK_TREE_NODE, "Initial Fluid Volumes:", NK_MAXIMIZED))
		{
			for (int i = 0; i < fluidConfig.initialVolumes.size();++i) {
				InitializationVolume& volume = fluidConfig.initialVolumes[i];
				if (nk_tree_push(ctx, NK_TREE_NODE, std::to_string(i).c_str(), NK_MAXIMIZED))
				{
					nk_layout_row_dynamic(ctx, 40, 3);
					nk_label(ctx, "Type:", NK_TEXT_LEFT);

					float incStep = 1.f/16;
					

					if (nk_option_label(ctx, "Box", volume.shapeType == ShapeType::Square)) {
						volume.shapeType = ShapeType::Square;
						while (volume.params.size() < 6) {
							volume.params.push_back(0);
						}
					}
						
					if (nk_option_label(ctx, "Ball", fluidConfig.initialVolumes[i].shapeType == ShapeType::Sphere)) {
						volume.shapeType = ShapeType::Sphere;
						while (volume.params.size() < 4) {
							volume.params.push_back(0);
						}
					}

					if (volume.shapeType == ShapeType::Square) {
						nk_layout_row_dynamic(ctx, 25, 1);
						nk_label(ctx, "Min Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, 25, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[0], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[1], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[2], 1, incStep, incPerPixel);

						nk_layout_row_dynamic(ctx, 25, 1);
						nk_label(ctx, "Max Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, 25, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[3], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[4], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[5], 1, incStep, incPerPixel);

						if (isFLIP) {
							nk_layout_row_dynamic(ctx, 25, 2);
							nk_label(ctx, "Phase:", NK_TEXT_LEFT);
							nk_property_int(ctx, "", 0, &volume.phase, fluidConfig.phaseCount, 1, incPerPixel);
						}
						nk_layout_row_dynamic(ctx, 25, 1);
						if (nk_button_label(ctx, "delete")) {

						}
					}
					
					else if (volume.shapeType == ShapeType::Sphere) {
						nk_layout_row_dynamic(ctx, 25, 1);
						nk_label(ctx, "Center Coordinate:", NK_TEXT_LEFT);
						nk_layout_row_dynamic(ctx, 25, 3);

						nk_property_float(ctx, "x:", 0, &volume.params[0], 1, incStep, incPerPixel);
						nk_property_float(ctx, "y:", 0, &volume.params[1], 1, incStep, incPerPixel);
						nk_property_float(ctx, "z:", 0, &volume.params[2], 1, incStep, incPerPixel);

						nk_layout_row_dynamic(ctx, 25, 2);
						nk_label(ctx, "Radius:", NK_TEXT_LEFT);
						nk_property_float(ctx, "r:", 0, &volume.params[3], 1, incStep, incPerPixel);
						if (isFLIP) {
							nk_layout_row_dynamic(ctx, 25, 2);
							nk_label(ctx, "Phase:", NK_TEXT_LEFT);
							nk_property_int(ctx, "", 0, &volume.phase, fluidConfig.phaseCount, 1, incPerPixel);
						}
						nk_layout_row_dynamic(ctx, 25, 1);
						if (nk_button_label(ctx, "delete")) {

						}
					}

					nk_tree_pop(ctx);
				}
			}
			

			nk_tree_pop(ctx);
		}
		nk_layout_row_dynamic(ctx, 25, 1);
		if (nk_button_label(ctx, "Add New Initial Volume")) {

		}

		nk_layout_row_begin(ctx, NK_STATIC, 25, 5);
		nk_layout_row_push(ctx, 45);
		if (nk_menu_begin_label(ctx, "MENU", NK_TEXT_LEFT, nk_vec2(nk_widget_width(ctx), 400)))
		{
			bool show_menu, show_app_about;
			static size_t prog = 40;
			static int slider = 10;
			static int check = nk_true;
			nk_layout_row_dynamic(ctx, 25, 1);
			if (nk_menu_item_label(ctx, "Hide", NK_TEXT_LEFT))
				show_menu = nk_false;
			if (nk_menu_item_label(ctx, "About", NK_TEXT_LEFT))
				show_app_about = nk_true;
			nk_progress(ctx, &prog, 100, NK_MODIFIABLE);
			nk_slider_int(ctx, 0, &slider, 16, 1);
			nk_checkbox_label(ctx, "check", &check);

			nk_menu_end(ctx);
		}

		enum { EASY, HARD };
		static int op = EASY;
		static int property = 20;
		nk_layout_row_static(ctx, 30, 80, 1);
		if (nk_button_label(ctx, "button"))
			fprintf(stdout, "button pressed\n");

		nk_layout_row_dynamic(ctx, 30, 2);
		if (nk_option_label(ctx, "easy", op == EASY)) op = EASY;
		if (nk_option_label(ctx, "hard", op == HARD)) op = HARD;

		

		nk_layout_row_dynamic(ctx, 25, 1);
		nk_property_int(ctx, "Compression:", 0, &property, 100, 10, 1);

		nk_layout_row_dynamic(ctx, 20, 1);
		nk_label(ctx, "background:", NK_TEXT_LEFT);
		nk_layout_row_dynamic(ctx, 25, 1);
		if (nk_combo_begin_color(ctx, nk_rgb_cf(bg), nk_vec2(nk_widget_width(ctx), 400))) {
			nk_layout_row_dynamic(ctx, 120, 1);
			bg = nk_color_picker(ctx, bg, NK_RGBA);
			nk_layout_row_dynamic(ctx, 25, 1);
			bg.r = nk_propertyf(ctx, "#R:", 0, bg.r, 1.0f, 0.01f, 0.005f);
			bg.g = nk_propertyf(ctx, "#G:", 0, bg.g, 1.0f, 0.01f, 0.005f);
			bg.b = nk_propertyf(ctx, "#B:", 0, bg.b, 1.0f, 0.01f, 0.005f);
			bg.a = nk_propertyf(ctx, "#A:", 0, bg.a, 1.0f, 0.01f, 0.005f);
			nk_combo_end(ctx);
		}
	}
	nk_end(ctx);
#define MAX_VERTEX_BUFFER 512 * 1024
#define MAX_ELEMENT_BUFFER 128 * 1024

	nk_glfw3_render(NK_ANTI_ALIASING_ON, MAX_VERTEX_BUFFER, MAX_ELEMENT_BUFFER);
}