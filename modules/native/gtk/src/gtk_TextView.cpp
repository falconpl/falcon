/**
 *  \file gtk_TextView.cpp
 */

#include "gtk_TextView.hpp"

#include "gtk_TextBuffer.hpp"
#include "gtk_TextIter.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void TextView::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_TextView = mod->addClass( "GtkTextView", &TextView::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkContainer" ) );
    c_TextView->getClassDef()->addInheritance( in );

    c_TextView->setWKS( true );
    c_TextView->getClassDef()->factory( &TextView::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_buffer",        &TextView::new_with_buffer },
    { "set_buffer",             &TextView::set_buffer },
    { "get_buffer",             &TextView::get_buffer },
    { "scroll_to_mark",         &TextView::scroll_to_mark },
    { "scroll_to_iter",         &TextView::scroll_to_iter },
    { "scroll_mark_onscreen",   &TextView::scroll_mark_onscreen },
    { "move_mark_onscreen",     &TextView::move_mark_onscreen },
    { "place_cursor_onscreen",  &TextView::place_cursor_onscreen },
    //{ "get_visible_rect",       &TextView::get_visible_rect },
    //{ "get_iter_location",      &TextView::get_iter_location },
    { "get_line_at_y",          &TextView::get_line_at_y },
    { "get_line_yrange",        &TextView::get_line_yrange },
    { "get_iter_at_location",   &TextView::get_iter_at_location },
    { "get_iter_at_position",   &TextView::get_iter_at_position },
    { "buffer_to_window_coords",&TextView::buffer_to_window_coords },
    { "window_to_buffer_coords",&TextView::window_to_buffer_coords },
    //{ "get_window",             &TextView::get_window },
    //{ "get_window_type",        &TextView::get_window_type },
    { "set_border_window_size", &TextView::set_border_window_size },
    { "get_border_window_size", &TextView::get_border_window_size },
    { "forward_display_line",   &TextView::forward_display_line },
    { "backward_display_line",  &TextView::backward_display_line },
    { "forward_display_line_end",&TextView::forward_display_line_end },
    { "backward_display_line_start",&TextView::backward_display_line_start },
    { "starts_display_line",    &TextView::starts_display_line },
    { "move_visually",          &TextView::move_visually },
    //{ "add_child_at_anchor",    &TextView::add_child_at_anchor },
    { "add_child_in_window",    &TextView::add_child_in_window },
    { "move_child",             &TextView::move_child },
    { "set_wrap_mode",          &TextView::set_wrap_mode },
    { "get_wrap_mode",          &TextView::get_wrap_mode },
    { "set_editable",           &TextView::set_editable },
    { "get_editable",           &TextView::get_editable },
    { "set_cursor_visible",     &TextView::set_cursor_visible },
    { "get_cursor_visible",     &TextView::get_cursor_visible },
    { "set_overwrite",          &TextView::set_overwrite },
    { "get_overwrite",          &TextView::get_overwrite },
    { "set_pixels_above_lines", &TextView::set_pixels_above_lines },
    { "get_pixels_above_lines", &TextView::get_pixels_above_lines },
    { "set_pixels_below_lines", &TextView::set_pixels_below_lines },
    { "get_pixels_below_lines", &TextView::get_pixels_below_lines },
    { "set_pixels_inside_wrap", &TextView::set_pixels_inside_wrap },
    { "get_pixels_inside_wrap", &TextView::get_pixels_inside_wrap },
    { "set_justification",      &TextView::set_justification },
    { "get_justification",      &TextView::get_justification },
    { "set_left_margin",        &TextView::set_left_margin },
    { "get_left_margin",        &TextView::get_left_margin },
    { "set_right_margin",       &TextView::set_right_margin },
    { "get_right_margin",       &TextView::get_right_margin },
    { "set_indent",             &TextView::set_indent },
    { "get_indent",             &TextView::get_indent },
    //{ "set_tabs",               &TextView::set_tabs },
    //{ "get_tabs",               &TextView::get_tabs },
    { "set_accepts_tab",        &TextView::set_accepts_tab },
    { "get_accepts_tab",        &TextView::get_accepts_tab },
    //{ "get_default_attributes", &TextView::get_default_attributes },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextView, meth->name, meth->cb );
}


TextView::TextView( const Falcon::CoreClass* gen, const GtkTextView* view )
    :
    Gtk::CoreGObject( gen, (GObject*) view )
{}


Falcon::CoreObject* TextView::factory( const Falcon::CoreClass* gen, void* view, bool )
{
    return new TextView( gen, (GtkTextView*) view );
}


/*#
    @class GtkTextView
    @brief Widget that displays a GtkTextBuffer

    You may wish to begin by reading the text widget conceptual overview which gives
    an overview of all the objects and data types related to the text widget and how
    they work together.
 */
FALCON_FUNC TextView::init( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GtkWidget* view = gtk_text_view_new();
    self->setObject( (GObject*) view );
}


/*#
    @method new_with_buffer GtkTextView
    @brief Creates a new GtkTextView widget displaying the buffer buffer.
    @param buffer a GtkTextBuffer (or nil)

    One buffer can be shared among many widgets. buffer may be nil to create a default
    buffer, in which case this function is equivalent to gtk_text_view_new(). The text
    view adds its own reference count to the buffer.
 */
FALCON_FUNC TextView::new_with_buffer( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[GtkTextBuffer]" );

    CoreGObject* o_buf = args.getCoreGObject( 0, false );
    GtkTextBuffer* buf = NULL;
    if ( o_buf )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !CoreObject_IS_DERIVED( o_buf, GtkTextBuffer ) )
            throw_inv_params( "[GtkTextBuffer]" );
#endif
        buf = (GtkTextBuffer*) o_buf->getObject();
    }
    GtkWidget* view = gtk_text_view_new_with_buffer( buf );
    vm->retval( new Gtk::TextView( vm->findWKI( "GtkTextView" )->asClass(),
            (GtkTextView*) view ) );
}


/*#
    @method set_buffer GtkTextView
    @brief Sets buffer as the buffer being displayed by text_view.
    @param buffer a GtkTextBuffer (or nil)
 */
FALCON_FUNC TextView::set_buffer( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[GtkTextBuffer]" );

    CoreGObject* o_buf = args.getCoreGObject( 0, false );
    GtkTextBuffer* buf = NULL;
    if ( o_buf )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !CoreObject_IS_DERIVED( o_buf, GtkTextBuffer ) )
            throw_inv_params( "[GtkTextBuffer]" );
#endif
        buf = (GtkTextBuffer*) o_buf->getObject();
    }
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_buffer( (GtkTextView*)_obj, buf );
}


/*#
    @method get_buffer GtkTextView
    @brief Returns the GtkTextBuffer being displayed by this text view.
    @return a GtkTextBuffer.
 */
FALCON_FUNC TextView::get_buffer( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkTextBuffer* buf = gtk_text_view_get_buffer( (GtkTextView*)_obj );
    vm->retval( new Gtk::TextBuffer( vm->findWKI( "GtkTextBuffer" )->asClass(), buf ) );
}


/*#
    @method scroll_to_mark GtkTextView
    @brief Scrolls text_view so that mark is on the screen in the position indicated by xalign and yalign.
    @param mark a GtkTextMark
    @param within_margin margin as a [0.0,0.5] fraction of screen size
    @param use_align (boolean) whether to use alignment arguments (if false, just get the mark onscreen)
    @param xalign horizontal alignment of mark within visible area
    @param yalign vertical alignment of mark within visible area

    An alignment of 0.0 indicates left or top, 1.0 indicates right or bottom, 0.5
    means center. If use_align is FALSE, the text scrolls the minimal distance to
    get the mark onscreen, possibly not scrolling at all. The effective screen for
    purposes of this function is reduced by a margin of size within_margin.
 */
FALCON_FUNC TextView::scroll_to_mark( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextMark,N,B,N,N" );

    CoreGObject* o_mk = args.getCoreGObject( 0 );
    gdouble within_margin = args.getNumeric( 1 );
    gboolean use_align = args.getBoolean( 2 );
    gdouble xalign = args.getNumeric( 3 );
    gdouble yalign = args.getNumeric( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_mk, GtkTextMark ) )
        throw_inv_params( "GtkTextMark,N,B,N,N" );
#endif
    GtkTextMark* mk = (GtkTextMark*) o_mk->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_scroll_to_mark( (GtkTextView*)_obj, mk, within_margin,
            use_align, xalign, yalign );
}


/*#
    @method scroll_to_iter GtkTextView
    @brief Scrolls text_view so that iter is on the screen in the position indicated by xalign and yalign.
    @param iter a GtkTextIter
    @param within_margin margin as a [0.0,0.5] fraction of screen size
    @param use_align (boolean) whether to use alignment arguments (if false, just get the mark onscreen)
    @param xalign horizontal alignment of mark within visible area
    @param yalign vertical alignment of mark within visible area
    @return true if scrolling occurred

    An alignment of 0.0 indicates left or top, 1.0 indicates right or bottom, 0.5
    means center. If use_align is FALSE, the text scrolls the minimal distance to
    get the mark onscreen, possibly not scrolling at all. The effective screen for
    purposes of this function is reduced by a margin of size within_margin.

    Note that this function uses the currently-computed height of the lines in the
    text buffer. Line heights are computed in an idle handler; so this function may
    not have the desired effect if it's called before the height computations.
    To avoid oddness, consider using scroll_to_mark() which saves a
    point to be scrolled to after line validation.
 */
FALCON_FUNC TextView::scroll_to_iter( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextIter,N,B,N,N" );

    CoreGObject* o_iter = args.getCoreGObject( 0 );
    gdouble within_margin = args.getNumeric( 1 );
    gboolean use_align = args.getBoolean( 2 );
    gdouble xalign = args.getNumeric( 3 );
    gdouble yalign = args.getNumeric( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,N,B,N,N" );
#endif
    GtkTextIter* iter = (GtkTextIter*) o_iter->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_scroll_to_iter( (GtkTextView*)_obj,
            iter, within_margin, use_align, xalign, yalign ) );
}


/*#
    @method scroll_mark_onscreen GtkTextView
    @brief Scrolls text_view the minimum distance such that mark is contained within the visible area of the widget.
    @param mark a mark in the buffer
 */
FALCON_FUNC TextView::scroll_mark_onscreen( VMARG )
{
    Item* i_mk = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mk || i_mk->isNil() || !i_mk->isObject()
        || !IS_DERIVED( i_mk, GtkTextMark ) )
        throw_inv_params( "GtkTextMark" );
#endif
    GtkTextMark* mk = (GtkTextMark*) COREGOBJECT( i_mk )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_scroll_mark_onscreen( (GtkTextView*)_obj, mk );
}


/*#
    @method move_mark_onscreen GtkTextView
    @brief Moves a mark within the buffer so that it's located within the currently-visible text area.
    @param mark a GtkTextMark
    @return true if the mark moved (wasn't already onscreen)
 */
FALCON_FUNC TextView::move_mark_onscreen( VMARG )
{
    Item* i_mk = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mk || i_mk->isNil() || !i_mk->isObject()
        || !IS_DERIVED( i_mk, GtkTextMark ) )
        throw_inv_params( "GtkTextMark" );
#endif
    GtkTextMark* mk = (GtkTextMark*) COREGOBJECT( i_mk )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_move_mark_onscreen( (GtkTextView*)_obj, mk ) );
}


/*#
    @method place_cursor_onscreen GtkTextView
    @brief Moves the cursor to the currently visible region of the buffer, it it isn't there already.
    @return true if the cursor had to be moved.
 */
FALCON_FUNC TextView::place_cursor_onscreen( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_place_cursor_onscreen( (GtkTextView*)_obj ) );
}


//FALCON_FUNC TextView::get_visible_rect( VMARG );

//FALCON_FUNC TextView::get_iter_location( VMARG );


/*#
    @method get_line_at_y GtkTextView
    @brief Gets the GtkTextIter at the start of the line containing the coordinate y.
    @param target_iter a GtkTextIter
    @param y a y coordinate (integer)
    @return top coordinate of the line

    y is in buffer coordinates, convert from window coordinates with
    gtk_text_view_window_to_buffer_coords(). If non-NULL, line_top will be filled
    with the coordinate of the top edge of the line.
 */
FALCON_FUNC TextView::get_line_at_y( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter )
        || !i_y || i_y->isNil() || !i_y->isInteger() )
        throw_inv_params( "GtkTextIter,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    gint line_top;
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_get_line_at_y( (GtkTextView*)_obj, iter, i_y->asInteger(), &line_top );
    vm->retval( line_top );
}


/*#
    @method get_line_yrange GtkTextView
    @brief Gets the y coordinate of the top of the line containing iter, and the height of the line.
    @param iter a GtkTextIter
    @return [ y coordinate, height ]

    The coordinate is a buffer coordinate; convert to window coordinates with
    buffer_to_window_coords().
 */
FALCON_FUNC TextView::get_line_yrange( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    gint y, height;
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_get_line_yrange( (GtkTextView*)_obj, iter, &y, &height );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( y );
    arr->append( height );
    vm->retval( arr );
}


/*#
    @method get_iter_at_location GtkTextView
    @brief Retrieves the iterator at buffer coordinates x and y.
    @param x x position, in buffer coordinates
    @param y y position, in buffer coordinates
    @return a GtkTextIter

    Buffer coordinates are coordinates for the entire buffer, not just the currently-displayed
    portion. If you have coordinates from an event, you have to convert those to buffer
    coordinates with window_to_buffer_coords().
 */
FALCON_FUNC TextView::get_iter_at_location( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || i_x->isNil() || !i_x->isInteger()
        || !i_y || i_y->isNil() || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTextIter* iter = (GtkTextIter*) memAlloc( sizeof( GtkTextIter ) );
    gtk_text_view_get_iter_at_location( (GtkTextView*)_obj, iter,
            i_x->asInteger(), i_y->asInteger() );
    vm->retval( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
}


/*#
    @method get_iter_at_position GtkTextView
    @brief Retrieves the iterator pointing to the character at buffer coordinates x and y.
    @param x x position, in buffer coordinates
    @param y y position, in buffer coordinates
    @return [ GtkTextIter, trailing ]. trailing is an integer indicating where in the grapheme the user clicked. It will either be zero, or the number of characters in the grapheme. 0 represents the trailing edge of the grapheme.

    Buffer coordinates are coordinates for the entire buffer, not just the
    currently-displayed portion. If you have coordinates from an event, you have
    to convert those to buffer coordinates with window_to_buffer_coords().

    Note that this is different from get_iter_at_location(), which returns cursor
    locations, i.e. positions between characters.
 */
FALCON_FUNC TextView::get_iter_at_position( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || i_x->isNil() || !i_x->isInteger()
        || !i_y || i_y->isNil() || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTextIter* iter = (GtkTextIter*) memAlloc( sizeof( GtkTextIter ) );
    gint trailing;

    gtk_text_view_get_iter_at_position( (GtkTextView*)_obj,
            iter, &trailing, i_x->asInteger(), i_y->asInteger() );

    CoreArray* arr = new CoreArray( 2 );
    arr->append( new Gtk::TextIter( vm->findWKI( "GtkTextIter" )->asClass(), iter ) );
    arr->append( trailing );
    vm->retval( arr );
}


/*#
    @method buffer_to_window_coords GtkTextView
    @brief Converts coordinate (buffer_x, buffer_y) to coordinates for the window win, and stores the result in (window_x, window_y).
    @param win a GtkTextWindowType except GTK_TEXT_WINDOW_PRIVATE
    @param buffer_x buffer x coordinate
    @param buffer_y buffer y coordinate
    @return [ window_x, window_y ]

    Note that you can't convert coordinates for a nonexisting window
    (see set_border_window_size()).
 */
FALCON_FUNC TextView::buffer_to_window_coords( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextWindowType,I,I" );

    gint win = args.getInteger( 0 );
    gint bx = args.getInteger( 1 );
    gint by = args.getInteger( 2 );

    MYSELF;
    GET_OBJ( self );
    gint wx, wy;
    gtk_text_view_buffer_to_window_coords( (GtkTextView*)_obj,
            (GtkTextWindowType) win, bx, by, &wx, &wy );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( wx );
    arr->append( wy );
    vm->retval( arr );
}


/*#
    @method window_to_buffer_coords GtkTextView
    @brief Converts coordinates on the window identified by win to buffer coordinates, storing the result in (buffer_x,buffer_y).
    @param win a GtkTextWindowType except GTK_TEXT_WINDOW_PRIVATE
    @param window_x window x coordinate
    @param window_y window y coordinate
    @return [ buffer_x, buffer_y ]

    Note that you can't convert coordinates for a nonexisting window
    (see set_border_window_size()).
 */
FALCON_FUNC TextView::window_to_buffer_coords( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkTextWindowType,I,I" );

    gint win = args.getInteger( 0 );
    gint wx = args.getInteger( 1 );
    gint wy = args.getInteger( 2 );

    MYSELF;
    GET_OBJ( self );
    gint bx, by;
    gtk_text_view_buffer_to_window_coords( (GtkTextView*)_obj,
            (GtkTextWindowType) win, wx, wy, &bx, &by );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( bx );
    arr->append( by );
    vm->retval( arr );
}


//FALCON_FUNC TextView::get_window( VMARG );

//FALCON_FUNC TextView::get_window_type( VMARG );


/*#
    @method set_border_window_size GtkTextView
    @brief Sets the width of GTK_TEXT_WINDOW_LEFT or GTK_TEXT_WINDOW_RIGHT, or the height of GTK_TEXT_WINDOW_TOP or GTK_TEXT_WINDOW_BOTTOM.
    @param type (GtkTextWindowType) window to affect
    @param size width or height of the window

    Automatically destroys the corresponding window if the size is set to 0, and
    creates the window if the size is set to non-zero. This function can only be
    used for the "border windows," it doesn't work with GTK_TEXT_WINDOW_WIDGET,
    GTK_TEXT_WINDOW_TEXT, or GTK_TEXT_WINDOW_PRIVATE.
 */
FALCON_FUNC TextView::set_border_window_size( VMARG )
{
    Item* i_win = vm->param( 0 );
    Item* i_sz = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_win || i_win->isNil() || !i_win->isInteger()
        || !i_sz || i_sz->isNil() || !i_sz->isInteger() )
        throw_inv_params( "I,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_border_window_size( (GtkTextView*)_obj,
            (GtkTextWindowType) i_win->asInteger(), i_sz->asInteger() );
}


/*#
    @method get_border_window_size GtkTextView
    @brief Gets the width of the specified border window.
    @param type (GtkTextWindowType) window to return size from
    @return width of window

    See set_border_window_size().
 */
FALCON_FUNC TextView::get_border_window_size( VMARG )
{
    Item* i_type = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_type || i_type->isNil() || !i_type->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_border_window_size( (GtkTextView*)_obj,
        (GtkTextWindowType) i_type->asInteger() ) );
}


/*#
    @method forward_display_line GtkTextView
    @brief Moves the given iter forward by one display (wrapped) line.
    @param iter a GtkTextIter
    @return true if iter was moved and is not on the end iterator

    A display line is different from a paragraph. Paragraphs are separated by newlines
    or other paragraph separator characters. Display lines are created by line-wrapping
    a paragraph. If wrapping is turned off, display lines and paragraphs will be the same.
    Display lines are divided differently for each view, since they depend on the view's
    width; paragraphs are the same in all views, since they depend on the contents of
    the GtkTextBuffer.
 */
FALCON_FUNC TextView::forward_display_line( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_forward_display_line( (GtkTextView*)_obj, iter ) );
}


/*#
    @method backward_display_line GtkTextView
    @brief Moves the given iter backward by one display (wrapped) line.
    @param iter a GtkTextIter
    @return true if iter was moved and is not on the end iterator

    A display line is different from a paragraph. Paragraphs are separated by newlines
    or other paragraph separator characters. Display lines are created by line-wrapping
    a paragraph. If wrapping is turned off, display lines and paragraphs will be the
    same. Display lines are divided differently for each view, since they depend on
    the view's width; paragraphs are the same in all views, since they depend on the
    contents of the GtkTextBuffer.
 */
FALCON_FUNC TextView::backward_display_line( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_backward_display_line( (GtkTextView*)_obj, iter ) );
}


/*#
    @method forward_display_line_end GtkTextView
    @brief Moves the given iter forward to the next display line end.
    @param iter a GtkTextIter
    @return true if iter was moved and is not on the end iterator

    A display line is different from a paragraph. Paragraphs are separated by newlines
    or other paragraph separator characters. Display lines are created by line-wrapping
    a paragraph. If wrapping is turned off, display lines and paragraphs will be the same.
    Display lines are divided differently for each view, since they depend on the view's
    width; paragraphs are the same in all views, since they depend on the contents of the
    GtkTextBuffer.
 */
FALCON_FUNC TextView::forward_display_line_end( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_forward_display_line_end( (GtkTextView*)_obj, iter ) );
}


/*#
    @method backward_display_line_start GtkTextView
    @brief Moves the given iter backward to the next display line start.
    @param iter a GtkTextIter
    @return true if iter was moved and is not on the end iterator

    A display line is different from a paragraph. Paragraphs are separated by newlines
    or other paragraph separator characters. Display lines are created by line-wrapping
    a paragraph. If wrapping is turned off, display lines and paragraphs will be the same.
    Display lines are divided differently for each view, since they depend on the view's
    width; paragraphs are the same in all views, since they depend on the contents of
    the GtkTextBuffer.
 */
FALCON_FUNC TextView::backward_display_line_start( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_backward_display_line_start( (GtkTextView*)_obj, iter ) );
}


/*#
    @method starts_display_line GtkTextView
    @brief Determines whether iter is at the start of a display line.
    @param iter a GtkTextIter
    @return true if iter begins a wrapped line

    See gtk_text_view_forward_display_line() for an explanation of display lines vs. paragraphs.
 */
FALCON_FUNC TextView::starts_display_line( VMARG )
{
    Item* i_iter = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_starts_display_line( (GtkTextView*)_obj, iter ) );
}


/*#
    @method move_visually GtkTextView
    @brief Move the iterator a given number of characters visually, treating it as the strong cursor position.
    @param iter a GtkTextIter
    @param count number of characters to move (negative moves left, positive moves right)
    @return true if iter moved and is not on the end iterator

    If count is positive, then the new strong cursor position will be count positions
    to the right of the old cursor position. If count is negative then the new strong
    cursor position will be count positions to the left of the old cursor position.

    In the presence of bi-directional text, the correspondence between logical and
    visual order will depend on the direction of the current run, and there may be
    jumps when the cursor is moved off of the end of a run.
 */
FALCON_FUNC TextView::move_visually( VMARG )
{
    Item* i_iter = vm->param( 0 );
    Item* i_cnt = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_iter || i_iter->isNil() || !i_iter->isObject()
        || !IS_DERIVED( i_iter, GtkTextIter )
        || !i_cnt || i_cnt->isNil() || !i_cnt->isInteger() )
        throw_inv_params( "GtkTextIter,I" );
#endif
    GtkTextIter* iter = (GtkTextIter*) COREGOBJECT( i_iter )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_move_visually( (GtkTextView*)_obj,
            iter, i_cnt->asInteger() ) );
}


//FALCON_FUNC TextView::add_child_at_anchor( VMARG );


/*#
    @method add_child_in_window GtkTextView
    @brief Adds a child at fixed coordinates in one of the text widget's windows.
    @param child a GtkWidget
    @param which_window (GtkTextWindowType) which window the child should appear in
    @param xpos X position of child in window coordinates
    @param ypos Y position of child in window coordinates

    The window must have nonzero size (see set_border_window_size()).
    Note that the child coordinates are given relative to the GdkWindow in question,
    and that these coordinates have no sane relationship to scrolling. When placing
    a child in GTK_TEXT_WINDOW_WIDGET, scrolling is irrelevant, the child floats
    above all scrollable areas. But when placing a child in one of the scrollable
    windows (border windows or text window), you'll need to compute the child's
    correct position in buffer coordinates any time scrolling occurs or buffer
    changes occur, and then call gtk_text_view_move_child() to update the child's
    position. Unfortunately there's no good way to detect that scrolling has occurred,
    using the current API; a possible hack would be to update all child positions
    when the scroll adjustments change or the text buffer changes. See bug 64518
    on bugzilla.gnome.org for status of fixing this issue.
 */
FALCON_FUNC TextView::add_child_in_window( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_win = vm->param( 1 );
    Item* i_xpos = vm->param( 2 );
    Item* i_ypos = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkWidget )
        || !i_win || i_win->isNil() || !i_win->isInteger()
        || !i_xpos || i_xpos->isNil() || !i_xpos->isInteger()
        || !i_ypos || i_ypos->isNil() || !i_ypos->isInteger() )
        throw_inv_params( "GtkWidget,GtkTextWindowType,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_add_child_in_window( (GtkTextView*)_obj, wdt,
        (GtkTextWindowType) i_win->asInteger(), i_xpos->asInteger(), i_ypos->asInteger() );
}


/*#
    @method move_child GtkTextView
    @brief Updates the position of a child, as for add_child_in_window().
    @param child child widget already added to the text view
    @param xpos new X position in window coordinates
    @param ypos new Y position in window coordinates
 */
FALCON_FUNC TextView::move_child( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_xpos = vm->param( 1 );
    Item* i_ypos = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkWidget )
        || !i_xpos || i_xpos->isNil() || !i_xpos->isInteger()
        || !i_ypos || i_ypos->isNil() || !i_ypos->isInteger() )
        throw_inv_params( "GtkWidget,I,I" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_move_child( (GtkTextView*)_obj, wdt,
            i_xpos->asInteger(), i_ypos->asInteger() );
}


/*#
    @method set_wrap_mode GtkTextView
    @brief Sets the line wrapping for the view.
    @param wrap_mode a GtkWrapMode
 */
FALCON_FUNC TextView::set_wrap_mode( VMARG )
{
    Item* i_mode = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_mode || i_mode->isNil() || !i_mode->isInteger() )
        throw_inv_params( "GtkWrapMode" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_wrap_mode( (GtkTextView*)_obj, (GtkWrapMode) i_mode->asInteger() );
}


/*#
    @method get_wrap_mode GtkTextView
    @brief Gets the line wrapping for the view.
    @return the line wrap setting
 */
FALCON_FUNC TextView::get_wrap_mode( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_wrap_mode( (GtkTextView*)_obj ) );
}


/*#
    @method set_editable GtkTextView
    @brief Sets the default editability of the GtkTextView.
    @param setting (boolean) whether it's editable
    You can override this default setting with tags in the buffer, using the "editable"
    attribute of tags.
 */
FALCON_FUNC TextView::set_editable( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_editable( (GtkTextView*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_editable GtkTextView
    @brief Returns the default editability of the GtkTextView.
    @return whether text is editable by default

    Tags in the buffer may override this setting for some ranges of text.
 */
FALCON_FUNC TextView::get_editable( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_get_editable( (GtkTextView*)_obj ) );
}


/*#
    @method set_cursor_visible GtkTextView
    @brief Toggles whether the insertion point is displayed.
    @param setting whether to show the insertion cursor

    A buffer with no editable text probably shouldn't have a visible cursor, so
    you may want to turn the cursor off.
 */
FALCON_FUNC TextView::set_cursor_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_cursor_visible( (GtkTextView*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_cursor_visible GtkTextView
    @brief Find out whether the cursor is being displayed.
    @return whether the insertion mark is visible
 */
FALCON_FUNC TextView::get_cursor_visible( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_get_cursor_visible( (GtkTextView*)_obj ) );
}


/*#
    @method set_overwrite GtkTextView
    @brief Changes the GtkTextView overwrite mode.
    @param true to turn on overwrite mode, false to turn it off
 */
FALCON_FUNC TextView::set_overwrite( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_overwrite( (GtkTextView*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_overwrite GtkTextView
    @brief Returns whether the GtkTextView is in overwrite mode or not.
    @return whether text_view is in overwrite mode or not.
 */
FALCON_FUNC TextView::get_overwrite( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_get_overwrite( (GtkTextView*)_obj ) );
}


/*#
    @method set_pixels_above_lines GtkTextView
    @brief Sets the default number of blank pixels above paragraphs in text_view.
    @param pixels_above_lines pixels above paragraphs

    Tags in the buffer for text_view may override the defaults.
 */
FALCON_FUNC TextView::set_pixels_above_lines( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_pixels_above_lines( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_pixels_above_lines GtkTextView
    @brief Gets the default number of pixels to put above paragraphs.
    @return default number of pixels above paragraphs
 */
FALCON_FUNC TextView::get_pixels_above_lines( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_pixels_above_lines( (GtkTextView*)_obj ) );
}


/*#
    @method set_pixels_below_lines GtkTextView
    @brief Sets the default number of pixels of blank space to put below paragraphs in text_view.
    @param pixels_below_lines pixels below paragraphs

    May be overridden by tags applied to text_view's buffer.
 */
FALCON_FUNC TextView::set_pixels_below_lines( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_pixels_below_lines( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_pixels_below_lines GtkTextView
    @brief Gets the value set by set_pixels_below_lines().
    @return default number of blank pixels below paragraphs
 */
FALCON_FUNC TextView::get_pixels_below_lines( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_pixels_below_lines( (GtkTextView*)_obj ) );
}


/*#
    @method set_pixels_inside_wrap GtkTextView
    @brief Sets the default number of pixels of blank space to leave between display/wrapped lines within a paragraph.
    @param pixels_inside_wrap default number of pixels between wrapped lines

    May be overridden by tags in text_view's buffer.
 */
FALCON_FUNC TextView::set_pixels_inside_wrap( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_pixels_inside_wrap( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_pixels_inside_wrap GtkTextView
    @brief Gets the value set by set_pixels_inside_wrap().
    @return default number of pixels of blank space between wrapped lines
 */
FALCON_FUNC TextView::get_pixels_inside_wrap( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_pixels_inside_wrap( (GtkTextView*)_obj ) );
}


/*#
    @method set_justification GtkTextView
    @brief Sets the default justification of text in text_view.
    @param justification (GtkJustification)

    Tags in the view's buffer may override the default.
 */
FALCON_FUNC TextView::set_justification( VMARG )
{
    Item* i_just = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_just || i_just->isNil() || !i_just->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_justification( (GtkTextView*)_obj, (GtkJustification) i_just->asInteger() );
}


/*#
    @method get_justification GtkTextView
    @brief Gets the default justification of paragraphs in text_view.
    @return default justification

    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::get_justification( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_text_view_get_justification( (GtkTextView*)_obj ) );
}


/*#
    @method set_left_margin GtkTextView
    @brief Sets the default left margin for text in text_view.
    @param left_margin left margin in pixels

    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::set_left_margin( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_left_margin( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_left_margin GtkTextView
    @brief Gets the default left margin size of paragraphs in the text_view.
    @return left margin in pixels
    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::get_left_margin( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_left_margin( (GtkTextView*)_obj ) );
}


/*#
    @method set_right_margin GtkTextView
    @brief Sets the default right margin for text in the text view.
    @param right_margin right margin in pixels

    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::set_right_margin( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_right_margin( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_right_margin GtkTextView
    @brief Sets the default right margin for text in the text view.
    @return right margin in pixels
    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::get_right_margin( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_right_margin( (GtkTextView*)_obj ) );
}


/*#
    @method set_indent GtkTextView
    @brief Sets the default indentation for paragraphs in text_view.
    @param indent indentation in pixels
    Tags in the buffer may override the default.
 */
FALCON_FUNC TextView::set_indent( VMARG )
{
    Item* i_num = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_num || i_num->isNil() || !i_num->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_indent( (GtkTextView*)_obj, i_num->asInteger() );
}


/*#
    @method get_indent GtkTextView
    @brief Gets the default indentation of paragraphs in text_view.
    @return number of pixels of indentation$

    Tags in the view's buffer may override the default. The indentation may be negative.
 */
FALCON_FUNC TextView::get_indent( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_text_view_get_indent( (GtkTextView*)_obj ) );
}


//FALCON_FUNC TextView::set_tabs( VMARG );

//FALCON_FUNC TextView::get_tabs( VMARG );


/*#
    @method set_accepts_tab GtkTextView
    @brief Sets the behavior of the text widget when the Tab key is pressed.
    @param accept_tag true if pressing the Tab key should insert a tab character, false, if pressing the Tab key should move the keyboard focus.
    If accepts_tab is true, a tab character is inserted. If accepts_tab is false
    the keyboard focus is moved to the next widget in the focus chain.
 */
FALCON_FUNC TextView::set_accepts_tab( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_accepts_tab( (GtkTextView*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_accepts_tab GtkTextView
    @brief Returns whether pressing the Tab key inserts a tab characters.
    @return true if pressing the Tab key inserts a tab character, false if pressing the Tab key moves the keyboard focus.
 */
FALCON_FUNC TextView::get_accepts_tab( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_get_accepts_tab( (GtkTextView*)_obj ) );
}


//FALCON_FUNC TextView::get_default_attributes( VMARG );


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
