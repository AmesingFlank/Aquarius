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
#define NK_IMPLEMENTATION
#define NK_GLFW_GL3_IMPLEMENTATION
#define NK_KEYSTATE_BASED_INPUT


#include <nuklear.h>

#include <nuklear_glfw_gl3.h>

#include "../Common/GpuCommons.h"

#include "../Rendering/WindowInfo.h"


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

#ifdef INCLUDE_STYLE
	/*set_style(ctx, THEME_WHITE);*/
	/*set_style(ctx, THEME_RED);*/
	/*set_style(ctx, THEME_BLUE);*/
	/*set_style(ctx, THEME_DARK);*/
#endif
	return ctx;
	
}

void drawUI(nk_context* ctx) {
	WindowInfo& windowInfo = WindowInfo::instance();
	int windowHeight = windowInfo.windowHeight;
	int windowWidth = windowInfo.windowWidth;

	nk_glfw3_new_frame();

	struct nk_colorf bg;
	bg.r = 0.10f, bg.g = 0.18f, bg.b = 0.24f, bg.a = 1.0f;

	if (nk_begin(ctx, "Configurations", nk_rect(windowWidth*0.02, windowWidth * 0.02, windowWidth * 0.2, windowHeight*0.9),
		NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
		NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
	{
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