/**
 *  \file gtk_TextView.cpp
 */

#include "gtk_TextView.hpp"

#include "gtk_TextBuffer.hpp"
#include "gtk_TextIter.hpp"

#include <gtk/gtk.h>


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
#if 0
    { "buffer_to_window_coords",        &TextView:: },
    { "window_to_buffer_coords",        &TextView:: },
    { "get_window",        &TextView:: },
    { "get_window_type",        &TextView:: },
    { "set_border_window_size",        &TextView:: },
    { "get_border_window_size",        &TextView:: },
    { "forward_display_line",        &TextView:: },
    { "backward_display_line",        &TextView:: },
    { "forward_display_line_end",        &TextView:: },
    { "backward_display_line_start",        &TextView:: },
    { "starts_display_line",        &TextView:: },
    { "move_visually",        &TextView:: },
    { "add_child_at_anchor",        &TextView:: },
    { "add_child_in_window",        &TextView:: },
    { "move_child",        &TextView:: },
    { "set_wrap_mode",        &TextView:: },
    { "get_wrap_mode",        &TextView:: },
    { "set_editable",        &TextView:: },
    { "get_editable",        &TextView:: },
    { "set_cursor_visible",        &TextView:: },
    { "get_cursor_visible",        &TextView:: },
    { "set_overwrite",        &TextView:: },
    { "get_overwrite",        &TextView:: },
    { "set_pixels_above_lines",        &TextView:: },
    { "get_pixels_above_lines",        &TextView:: },
    { "set_pixels_below_lines",        &TextView:: },
    { "get_pixels_below_lines",        &TextView:: },
    { "set_pixels_inside_wrap",        &TextView:: },
    { "get_pixels_inside_wrap",        &TextView:: },
    { "set_justification",        &TextView:: },
    { "get_justification",        &TextView:: },
    { "set_left_margin",        &TextView:: },
    { "get_left_margin",        &TextView:: },
    { "set_right_margin",        &TextView:: },
    { "get_right_margin",        &TextView:: },
    { "set_indent",        &TextView:: },
    { "get_indent",        &TextView:: },
    { "set_tabs",        &TextView:: },
    { "get_tabs",        &TextView:: },
    { "set_accepts_tab",        &TextView:: },
    { "get_accepts_tab",        &TextView:: },
    { "get_default_attributes",        &TextView:: },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_TextView, meth->name, meth->cb );
}


TextView::TextView( const Falcon::CoreClass* gen, const GtkTextView* view )
    :
    Gtk::CoreGObject( gen )
{
    if ( view )
        setUserData( new GData( Gtk::internal_add_slot( (GObject*) view ) ) );
}


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
    Gtk::internal_add_slot( (GObject*) view );
    self->setUserData( new GData( (GObject*) view ) );
}


/*#
    @method new_with_buffer
    @brief Creates a new GtkTextView widget displaying the buffer buffer.
    @param buffer a GtkTextBuffer (or nil)

    One buffer can be shared among many widgets. buffer may be nil to create a default
    buffer, in which case this function is equivalent to gtk_text_view_new(). The text
    view adds its own reference count to the buffer.
 */
FALCON_FUNC TextView::new_with_buffer( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[GtkTextBuffer]" );

    CoreObject* o_buf = args.getObject( 0, false );
    GtkTextBuffer* buf = NULL;
    if ( o_buf )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !CoreObject_IS_DERIVED( o_buf, GtkTextBuffer ) )
            throw_inv_params( "[GtkTextBuffer]" );
#endif
        buf = (GtkTextBuffer*)((GData*)o_buf->getUserData())->obj();
    }
    GtkWidget* view = gtk_text_view_new_with_buffer( buf );
    vm->retval( new Gtk::TextView( vm->findWKI( "GtkTextView" )->asClass(),
            (GtkTextView*) view ) );
}


/*#
    @method set_buffer
    @brief Sets buffer as the buffer being displayed by text_view.
    @param buffer a GtkTextBuffer (or nil)
 */
FALCON_FUNC TextView::set_buffer( VMARG )
{
    Gtk::ArgCheck0 args( vm, "[GtkTextBuffer]" );

    CoreObject* o_buf = args.getObject( 0, false );
    GtkTextBuffer* buf = NULL;
    if ( o_buf )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !CoreObject_IS_DERIVED( o_buf, GtkTextBuffer ) )
            throw_inv_params( "[GtkTextBuffer]" );
#endif
        buf = (GtkTextBuffer*)((GData*)o_buf->getUserData())->obj();
    }
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_set_buffer( (GtkTextView*)_obj, buf );
}


/*#
    @method get_buffer
    @brief Returns the GtkTextBuffer being displayed by this text view.
    @return a GtkTextBuffer.
 */
FALCON_FUNC TextView::get_buffer( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkTextBuffer* buf = gtk_text_view_get_buffer( (GtkTextView*)_obj );
    vm->retval( new Gtk::TextBuffer( vm->findWKI( "GtkTextBuffer" )->asClass(), buf ) );
}


/*#
    @method scroll_to_mark
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

    CoreObject* o_mk = args.getObject( 0 );
    gdouble within_margin = args.getNumeric( 1 );
    gboolean use_align = args.getBoolean( 2 );
    gdouble xalign = args.getNumeric( 3 );
    gdouble yalign = args.getNumeric( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_mk, GtkTextMark ) )
        throw_inv_params( "GtkTextMark,N,B,N,N" );
#endif
    GtkTextMark* mk = (GtkTextMark*)((GData*)o_mk->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_scroll_to_mark( (GtkTextView*)_obj, mk, within_margin,
            use_align, xalign, yalign );
}


/*#
    @method scroll_to_iter
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

    CoreObject* o_iter = args.getObject( 0 );
    gdouble within_margin = args.getNumeric( 1 );
    gboolean use_align = args.getBoolean( 2 );
    gdouble xalign = args.getNumeric( 3 );
    gdouble yalign = args.getNumeric( 4 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_iter, GtkTextIter ) )
        throw_inv_params( "GtkTextIter,N,B,N,N" );
#endif
    GtkTextIter* iter = (GtkTextIter*)((GData*)o_iter->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_scroll_to_iter( (GtkTextView*)_obj,
            iter, within_margin, use_align, xalign, yalign ) );
}


/*#
    @method scroll_mark_onscreen
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
    GtkTextMark* mk = (GtkTextMark*)((GData*)i_mk->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_scroll_mark_onscreen( (GtkTextView*)_obj, mk );
}


/*#
    @method move_mark_onscreen
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
    GtkTextMark* mk = (GtkTextMark*)((GData*)i_mk->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_move_mark_onscreen( (GtkTextView*)_obj, mk ) );
}


/*#
    @method place_cursor_onscreen
    @brief Moves the cursor to the currently visible region of the buffer, it it isn't there already.
    @return true if the cursor had to be moved.
 */
FALCON_FUNC TextView::place_cursor_onscreen( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_text_view_place_cursor_onscreen( (GtkTextView*)_obj ) );
}


//FALCON_FUNC TextView::get_visible_rect( VMARG );

//FALCON_FUNC TextView::get_iter_location( VMARG );


/*#
    @method get_line_at_y
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
    GtkTextIter* iter = (GtkTextIter*)((GData*)i_iter->asObject()->getUserData())->obj();
    gint line_top;
    MYSELF;
    GET_OBJ( self );
    gtk_text_view_get_line_at_y( (GtkTextView*)_obj, iter, i_y->asInteger(), &line_top );
    vm->retval( line_top );
}


/*#
    @method get_line_yrange
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
    GtkTextIter* iter = (GtkTextIter*)((GData*)i_iter->asObject()->getUserData())->obj();
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
    @method get_iter_at_location
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
    @method get_iter_at_position
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

#if 0
FALCON_FUNC TextView::buffer_to_window_coords( VMARG );

FALCON_FUNC TextView::window_to_buffer_coords( VMARG );

FALCON_FUNC TextView::get_window( VMARG );

FALCON_FUNC TextView::get_window_type( VMARG );

FALCON_FUNC TextView::set_border_window_size( VMARG );

FALCON_FUNC TextView::get_border_window_size( VMARG );

FALCON_FUNC TextView::forward_display_line( VMARG );

FALCON_FUNC TextView::backward_display_line( VMARG );

FALCON_FUNC TextView::forward_display_line_end( VMARG );

FALCON_FUNC TextView::backward_display_line_start( VMARG );

FALCON_FUNC TextView::starts_display_line( VMARG );

FALCON_FUNC TextView::move_visually( VMARG );

FALCON_FUNC TextView::add_child_at_anchor( VMARG );

FALCON_FUNC TextView::add_child_in_window( VMARG );

FALCON_FUNC TextView::move_child( VMARG );

FALCON_FUNC TextView::set_wrap_mode( VMARG );

FALCON_FUNC TextView::get_wrap_mode( VMARG );

FALCON_FUNC TextView::set_editable( VMARG );

FALCON_FUNC TextView::get_editable( VMARG );

FALCON_FUNC TextView::set_cursor_visible( VMARG );

FALCON_FUNC TextView::get_cursor_visible( VMARG );

FALCON_FUNC TextView::set_overwrite( VMARG );

FALCON_FUNC TextView::get_overwrite( VMARG );

FALCON_FUNC TextView::set_pixels_above_lines( VMARG );

FALCON_FUNC TextView::get_pixels_above_lines( VMARG );

FALCON_FUNC TextView::set_pixels_below_lines( VMARG );

FALCON_FUNC TextView::get_pixels_below_lines( VMARG );

FALCON_FUNC TextView::set_pixels_inside_wrap( VMARG );

FALCON_FUNC TextView::get_pixels_inside_wrap( VMARG );

FALCON_FUNC TextView::set_justification( VMARG );

FALCON_FUNC TextView::get_justification( VMARG );

FALCON_FUNC TextView::set_left_margin( VMARG );

FALCON_FUNC TextView::get_left_margin( VMARG );

FALCON_FUNC TextView::set_right_margin( VMARG );

FALCON_FUNC TextView::get_right_margin( VMARG );

FALCON_FUNC TextView::set_indent( VMARG );

FALCON_FUNC TextView::get_indent( VMARG );

FALCON_FUNC TextView::set_tabs( VMARG );

FALCON_FUNC TextView::get_tabs( VMARG );

FALCON_FUNC TextView::set_accepts_tab( VMARG );

FALCON_FUNC TextView::get_accepts_tab( VMARG );

FALCON_FUNC TextView::get_default_attributes( VMARG );
#endif

} // Gtk
} // Falcon
