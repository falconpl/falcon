/**
 *  \file gtk_CellRenderer.cpp
 */

#include "gtk_CellRenderer.hpp"

#include "gdk_Event.hpp"
#include "gdk_Rectangle.hpp"
#include "gdk_Window.hpp"

#include "gtk_CellEditable.hpp"
#include "gtk_Widget.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRenderer::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRenderer = mod->addClass( "GtkCellRenderer", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkObject" ) );
    c_CellRenderer->getClassDef()->addInheritance( in );

    //c_CellRenderer->setWKS( true );
    c_CellRenderer->getClassDef()->factory( &CellRenderer::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_editing_canceled",    &CellRenderer::signal_editing_canceled },
    { "signal_editing_started",     &CellRenderer::signal_editing_started },
    { "get_size",                   &CellRenderer::get_size },
    { "render",                     &CellRenderer::render },
    { "activate",                   &CellRenderer::activate },
    { "start_editing",              &CellRenderer::start_editing },
#if 0 // deprecated
    { "editing_canceled",           &CellRenderer::editing_canceled },
#endif
    { "stop_editing",               &CellRenderer::stop_editing },
    { "get_fixed_size",             &CellRenderer::get_fixed_size },
    { "set_fixed_size",             &CellRenderer::set_fixed_size },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "get_visible",                &CellRenderer::get_visible },
    { "set_visible",                &CellRenderer::set_visible },
    { "get_sensitive",              &CellRenderer::get_sensitive },
    { "set_sensitive",              &CellRenderer::set_sensitive },
    { "get_alignment",              &CellRenderer::get_alignment },
    { "set_alignment",              &CellRenderer::set_alignment },
    { "get_padding",                &CellRenderer::get_padding },
    { "set_padding",                &CellRenderer::set_padding },
#endif // GTK_CHECK_VERSION( 2, 18, 0 )
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_CellRenderer, meth->name, meth->cb );
}


CellRenderer::CellRenderer( const Falcon::CoreClass* gen, const GtkCellRenderer* entry )
    :
    Gtk::CoreGObject( gen, (GObject*) entry )
{}


Falcon::CoreObject* CellRenderer::factory( const Falcon::CoreClass* gen, void* entry, bool )
{
    return new CellRenderer( gen, (GtkCellRenderer*) entry );
}


/*#
    @class GtkCellRenderer
    @brief An object for rendering a single cell on a GdkDrawable

    The GtkCellRenderer is a base class of a set of objects used for rendering a
    cell to a GdkDrawable. These objects are used primarily by the GtkTreeView
    widget, though they aren't tied to them in any specific way. It is worth
    noting that GtkCellRenderer is not a GtkWidget and cannot be treated as such.

    The primary use of a GtkCellRenderer is for drawing a certain graphical
    elements on a GdkDrawable. Typically, one cell renderer is used to draw many
    cells on the screen. To this extent, it isn't expected that a CellRenderer
    keep any permanent state around. Instead, any state is set just prior to use
    using GObjects property system. Then, the cell is measured using
    gtk_cell_renderer_get_size(). Finally, the cell is rendered in the correct
    location using gtk_cell_renderer_render().

    There are a number of rules that must be followed when writing a new
    GtkCellRenderer. First and formost, it's important that a certain set of
    properties will always yield a cell renderer of the same size, barring a
    GtkStyle change. The GtkCellRenderer also has a number of generic properties
    that are expected to be honored by all children.

    Beyond merely rendering a cell, cell renderers can optionally provide active
    user interface elements. A cell renderer can be activatable like
    GtkCellRendererToggle, which toggles when it gets activated by a mouse click,
    or it can be editable like GtkCellRendererText, which allows the user to
    edit the text using a GtkEntry. To make a cell renderer activatable or
    editable, you have to implement the activate or start_editing virtual
    functions, respectively.
 */


/*#
    @method signal_editing_canceled GtkCellRenderer
    @brief This signal gets emitted when the user cancels the process of editing a cell.

    For example, an editable cell renderer could be written to cancel editing
    when the user presses Escape.

    See also: gtk_cell_renderer_stop_editing().
 */
FALCON_FUNC CellRenderer::signal_editing_canceled( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "editing_canceled", (void*) &CellRenderer::on_editing_canceled, vm );
}


void CellRenderer::on_editing_canceled( GtkCellRenderer* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "editing_canceled", "on_editing_canceled", (VMachine*)_vm );
}


/*#
    @method signal_editing_started GtkCellRenderer
    @brief This signal gets emitted when a cell starts to be edited.

    The intended use of this signal is to do special setup on editable,
    e.g. adding a GtkEntryCompletion or setting up additional columns in a
    GtkComboBox.

    [...]
 */
FALCON_FUNC CellRenderer::signal_editing_started( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "editing_started", (void*) &CellRenderer::on_editing_started, vm );
}


void CellRenderer::on_editing_started( GtkCellRenderer* obj,
                                       GtkCellEditable* editable, gchar* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "editing_started", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkCellEditable" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_editing_started", it ) )
            {
                printf(
                "[GtkCellRenderer::on_editing_started] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new Gtk::CellEditable( wki->asClass(), editable ) );
        vm->pushParam( UTF8String( path ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method get_size GtkCellRenderer
    @brief Obtains the width and height needed to render the cell.
    @param widget the widget the renderer is rendering to
    @param cell_area The area a cell will be allocated (GdkRectangle), or nil.
    @return an array ( xoffset, yoffset, width, height ).

    Used by view widgets to determine the appropriate size for the cell_area
    passed to gtk_cell_renderer_render().

    Please note that the values set in width and height, as well as those in
    x_offset and y_offset are inclusive of the xpad and ypad properties.
 */
FALCON_FUNC CellRenderer::get_size( VMARG )
{
    Item* i_wdt = vm->param( 0 );
    Item* i_area = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_area || !( i_area->isNil() || ( i_area->isObject()
        && IS_DERIVED( i_area, GtkWidget ) ) ) )
        throw_inv_params( "GtkWidget,[GdkRectangle]" );
#endif
    gint x, y, w, h;
    gtk_cell_renderer_get_size( GET_CELLRENDERER( vm->self() ),
                                GET_WIDGET( *i_wdt ),
                                i_area->isNil() ? NULL : GET_RECTANGLE( *i_area ),
                                &x, &y, &w, &h );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( x );
    arr->append( y );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method render GtkCellRenderer
    @brief Invokes the virtual render function of the GtkCellRenderer.
    @param window a GdkDrawable to draw to
    @param widget the widget owning window
    @param background_area entire cell area (GdkRectangle) (including tree expanders and maybe padding on the sides)
    @param cell_area area normally rendered by a cell renderer (GdkRectangle)
    @param expose_area area that actually needs updating (GdkRectangle)
    @param flags flags that affect rendering (GtkCellRendererState)

    The three passed-in rectangles are areas of window. Most renderers will draw
    within cell_area; the xalign, yalign, xpad, and ypad fields of the
    GtkCellRenderer should be honored with respect to cell_area. background_area
    includes the blank space around the cell, and also the area containing the
    tree expander; so the background_area rectangles for all cells tile to cover
    the entire window. expose_area is a clip rectangle.
 */
FALCON_FUNC CellRenderer::render( VMARG )
{
    Item* i_win = vm->param( 0 );
    Item* i_wdt = vm->param( 1 );
    Item* i_back_area = vm->param( 2 );
    Item* i_cell_area = vm->param( 3 );
    Item* i_expo_area = vm->param( 4 );
    Item* i_flags = vm->param( 5 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_win || !i_win->isObject() || !IS_DERIVED( i_win, GdkWindow )
        || !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_back_area || !i_back_area->isObject() || !IS_DERIVED( i_back_area, GdkRectangle )
        || !i_cell_area || !i_cell_area->isObject() || !IS_DERIVED( i_cell_area, GdkRectangle )
        || !i_expo_area || !i_expo_area->isObject() || !IS_DERIVED( i_expo_area, GdkRectangle )
        || !i_flags || !i_flags->isInteger() )
        throw_inv_params( "GdkWindow,GtkWidget,GdkRectangle,"
                          "GdkRectangle,GdkRectangle,GtkCellRendererState" );
#endif
    gtk_cell_renderer_render( GET_CELLRENDERER( vm->self() ),
                              GET_GDKWINDOW( *i_win ),
                              GET_WIDGET( *i_wdt ),
                              GET_RECTANGLE( *i_back_area ),
                              GET_RECTANGLE( *i_cell_area ),
                              GET_RECTANGLE( *i_expo_area ),
                              (GtkCellRendererState) i_flags->asInteger() );
}


/*#
    @method activate GtkCellRenderer
    @brief Passes an activate event to the cell renderer for possible processing.
    @param event a GdkEvent
    @param widget widget that received the event (GtkWidget).
    @param path widget-dependent string representation of the event location; e.g. for GtkTreeView, a string representation of GtkTreePath
    @param background_area background area as passed to gtk_cell_renderer_render() (GdkRectangle).
    @param cell_area cell area as passed to gtk_cell_renderer_render() (GdkRectangle)
    @param flags render flags (GtkCellRendererState)
    @return TRUE if the event was consumed/handled

    Some cell renderers may use events; for example, GtkCellRendererToggle
    toggles when it gets a mouse click.
 */
FALCON_FUNC CellRenderer::activate( VMARG )
{
    Item* i_ev = vm->param( 0 );
    Item* i_wdt = vm->param( 1 );
    Item* i_path = vm->param( 2 );
    Item* i_back_area = vm->param( 3 );
    Item* i_cell_area = vm->param( 4 );
    Item* i_flags = vm->param( 5 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !i_ev->isObject() || !IS_DERIVED( i_ev, GdkEvent )
        || !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_path || !i_path->isString()
        || !i_back_area || !i_back_area->isObject() || !IS_DERIVED( i_back_area, GdkRectangle )
        || !i_cell_area || !i_cell_area->isObject() || !IS_DERIVED( i_cell_area, GdkRectangle )
        || !i_flags || !i_flags->isInteger() )
        throw_inv_params( "GdkEvent,GtkWidget,S,GdkRectangle,"
                          "GdkRectangle,GtkCellRendererState" );
#endif
    AutoCString path( i_path->asString() );
    vm->retval( (bool)
    gtk_cell_renderer_activate( GET_CELLRENDERER( vm->self() ),
                                GET_EVENT( *i_ev ),
                                GET_WIDGET( *i_wdt ),
                                path.c_str(),
                                GET_RECTANGLE( *i_back_area ),
                                GET_RECTANGLE( *i_cell_area ),
                                (GtkCellRendererState) i_flags->asInteger() ) );
}


/*#
    @method start_editing GtkCellRenderer
    @brief Passes an activate event to the cell renderer for possible processing.
    @param event a GdkEvent
    @param widget widget that received the event (GtkWidget).
    @param path widget-dependent string representation of the event location; e.g. for GtkTreeView, a string representation of GtkTreePath
    @param background_area background area as passed to gtk_cell_renderer_render() (GdkRectangle).
    @param cell_area cell area as passed to gtk_cell_renderer_render() (GdkRectangle)
    @param flags render flags (GtkCellRendererState)
    @return A new GtkCellEditable, or NULL

 */
FALCON_FUNC CellRenderer::start_editing( VMARG )
{
    Item* i_ev = vm->param( 0 );
    Item* i_wdt = vm->param( 1 );
    Item* i_path = vm->param( 2 );
    Item* i_back_area = vm->param( 3 );
    Item* i_cell_area = vm->param( 4 );
    Item* i_flags = vm->param( 5 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !i_ev->isObject() || !IS_DERIVED( i_ev, GdkEvent )
        || !i_wdt || !i_wdt->isObject() || !IS_DERIVED( i_wdt, GtkWidget )
        || !i_path || !i_path->isString()
        || !i_back_area || !i_back_area->isObject() || !IS_DERIVED( i_back_area, GdkRectangle )
        || !i_cell_area || !i_cell_area->isObject() || !IS_DERIVED( i_cell_area, GdkRectangle )
        || !i_flags || !i_flags->isInteger() )
        throw_inv_params( "GdkEvent,GtkWidget,S,GdkRectangle,"
                          "GdkRectangle,GtkCellRendererState" );
#endif
    AutoCString path( i_path->asString() );
    GtkCellEditable* editable =
    gtk_cell_renderer_start_editing( GET_CELLRENDERER( vm->self() ),
                                     GET_EVENT( *i_ev ),
                                     GET_WIDGET( *i_wdt ),
                                     path.c_str(),
                                     GET_RECTANGLE( *i_back_area ),
                                     GET_RECTANGLE( *i_cell_area ),
                                     (GtkCellRendererState) i_flags->asInteger() );
    if ( editable )
        vm->retval( new Gtk::CellEditable( vm->findWKI( "GtkCellEditable" )->asClass(),
                                           editable ) );
    else
        vm->retnil();
}


#if 0 // deprecated
FALCON_FUNC CellRenderer::editing_canceled( VMARG );
#endif


/*#
    @method stop_editing GtkCellRenderer
    @brief Informs the cell renderer that the editing is stopped.
    @param canceled TRUE if the editing has been canceled

    If canceled is TRUE, the cell renderer will emit the "editing-canceled" signal.

    This function should be called by cell renderer implementations in
    response to the "editing-done" signal of GtkCellEditable.
 */
FALCON_FUNC CellRenderer::stop_editing( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_stop_editing( GET_CELLRENDERER( vm->self() ),
                                    (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_fixed_size GtkCellRenderer
    @brief Fills in width and height with the appropriate size of cell.
    @return an array [ fixed width of the cell, fixed height of the cell ].
 */
FALCON_FUNC CellRenderer::get_fixed_size( VMARG )
{
    NO_ARGS
    gint w, h;
    gtk_cell_renderer_get_fixed_size( GET_CELLRENDERER( vm->self() ), &w, &h );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method set_fixed_size GtkCellRenderer
    @brief Sets the renderer size to be explicit, independent of the properties set.
    @param width the width of the cell renderer, or -1
    @param height the height of the cell renderer, or -1
 */
FALCON_FUNC CellRenderer::set_fixed_size( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || !i_w->isInteger()
        || !i_h || !i_h->isInteger() )
        throw_inv_params( "I,I" );
#endif
    gtk_cell_renderer_set_fixed_size( GET_CELLRENDERER( vm->self() ),
                                      i_w->asInteger(),
                                      i_h->asInteger() );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method get_visible GtkCellRenderer
    @brief Returns the cell renderer's visibility.
    @return TRUE if the cell renderer is visible
 */
FALCON_FUNC CellRenderer::get_visible( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_cell_renderer_get_visible( GET_CELLRENDERER( vm->self() ) ) );
}


/*#
    @method set_visible GtkCellRenderer
    @brief Sets the cell renderer's visibility.
    @param visible the visibility of the cell
 */
FALCON_FUNC CellRenderer::set_visible( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_set_visible( GET_CELLRENDERER( vm->self() ),
                                   (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_sensitive GtkCellRenderer
    @brief Returns the cell renderer's sensitivity.
    @return TRUE if the cell renderer is sensitive
 */
FALCON_FUNC CellRenderer::get_sensitive( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_cell_renderer_get_sensitive( GET_CELLRENDERER( vm->self() ) ) );
}


/*#
    @method set_sensitive GtkCellRenderer
    @brief Sets the cell renderer's sensitivity.
    @param sensitive the sensitivity of the cell
 */
FALCON_FUNC CellRenderer::set_sensitive( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_set_sensitive( GET_CELLRENDERER( vm->self() ),
                                     (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_alignment GtkCellRenderer
    @brief Returns xalign and yalign of the cell.
    @return an array [ x alignment, y alignment ]
 */
FALCON_FUNC CellRenderer::get_alignment( VMARG )
{
    NO_ARGS
    gfloat x, y;
    gtk_cell_renderer_get_alignment( GET_CELLRENDERER( vm->self() ), &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( (numeric) x );
    arr->append( (numeric) y );
    vm->retval( arr );
}


/*#
    @method set_alignment GtkCellRenderer
    @brief Sets the renderer's alignment within its available space.
    @param x the x alignment of the cell renderer
    @param y the y alignment of the cell renderer
 */
FALCON_FUNC CellRenderer::set_alignment( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isOrdinal()
        || !i_y || !i_y->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    gtk_cell_renderer_set_alignment( GET_CELLRENDERER( vm->self() ),
                                     i_x->forceNumeric(),
                                     i_y->forceNumeric() );
}


/*#
    @method get_padding GtkCellRenderer
    @brief Returns xpad and ypad of the cell.
    @return an array ( x padding, y padding )
 */
FALCON_FUNC CellRenderer::get_padding( VMARG )
{
    NO_ARGS
    gint x, y;
    gtk_cell_renderer_get_padding( GET_CELLRENDERER( vm->self() ), &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


/*#
    @method set_padding GtkCellRenderer
    @brief Sets the renderer's padding.
    @param xpad the x padding of the cell renderer
    @param ypad the y padding of the cell renderer
 */
FALCON_FUNC CellRenderer::set_padding( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isInteger()
        || !i_y || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    gtk_cell_renderer_set_padding( GET_CELLRENDERER( vm->self() ),
                                   i_x->asInteger(),
                                   i_y->asInteger() );
}
#endif // GTK_CHECK_VERSION( 2, 18, 0 )


} // Gtk
} // Falcon
