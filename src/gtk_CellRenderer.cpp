/**
 *  \file gtk_CellRenderer.cpp
 */

#include "gtk_CellRenderer.hpp"

#include "gdk_Rectangle.hpp"


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
#if 0 // todo
    { "signal_editing_started",     &CellRenderer::signal_editing_started },
#endif
    { "get_size",                   &CellRenderer::get_size },
#if 0 // todo
    { "render",                     &CellRenderer::render },
    { "activate",                   &CellRenderer::activate },
    { "start_editing",              &CellRenderer::start_editing },
    { "editing_canceled",           &CellRenderer::editing_canceled },
    { "stop_editing",               &CellRenderer::stop_editing },
    { "get_fixed_size",             &CellRenderer::get_fixed_size },
    { "set_fixed_size",             &CellRenderer::set_fixed_size },
    { "get_visible",                &CellRenderer::get_visible },
    { "set_visible",                &CellRenderer::set_visible },
    { "get_sensitive",              &CellRenderer::get_sensitive },
    { "set_sensitive",              &CellRenderer::set_sensitive },
    { "get_alignment",              &CellRenderer::get_alignment },
    { "set_alignment",              &CellRenderer::set_alignment },
    { "get_padding",                &CellRenderer::get_padding },
    { "set_padding",                &CellRenderer::set_padding },
#endif
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
    @method signal_editing_canceled
    @brief This signal gets emitted when the user cancels the process of editing a cell.

    For example, an editable cell renderer could be written to cancel editing
    when the user presses Escape.

    See also: gtk_cell_renderer_stop_editing().
 */
FALCON_FUNC CellRenderer::signal_editing_canceled( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "editing_canceled", (void*) &CellRenderer::on_editing_canceled, vm );
}


void CellRenderer::on_editing_canceled( GtkCellRenderer* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "editing_canceled", "on_editing_canceled", (VMachine*)_vm );
}

#if 0 // todo (GtkCellEditable)
/*#
    @method signal_editing_started
    @brief This signal gets emitted when a cell starts to be edited.

    The intended use of this signal is to do special setup on editable,
    e.g. adding a GtkEntryCompletion or setting up additional columns in a
    GtkComboBox.

    [...]
 */
FALCON_FUNC CellRenderer::signal_editing_started( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "editing_started", (void*) &CellRenderer::on_editing_started, vm );
}


//void CellRenderer::on_editing_started( GtkCellRenderer*, GtkCellEditable*, gchar*, gpointer );
#endif

/*#
    @method get_size
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
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getGObject();
    GdkRectangle* area = i_area->isNil() ? NULL
            : dyncast<Gdk::Rectangle*>( i_area->asObjectSafe() )->getRectangle();
    MYSELF;
    GET_OBJ( self );
    gint x, y, w, h;
    gtk_cell_renderer_get_size( (GtkCellRenderer*)_obj, wdt, area, &x, &y, &w, &h );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( x );
    arr->append( y );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}

#if 0
/*#
    @method render
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

}


FALCON_FUNC CellRenderer::activate( VMARG );
FALCON_FUNC CellRenderer::start_editing( VMARG );
FALCON_FUNC CellRenderer::editing_canceled( VMARG );
FALCON_FUNC CellRenderer::stop_editing( VMARG );
FALCON_FUNC CellRenderer::get_fixed_size( VMARG );
FALCON_FUNC CellRenderer::set_fixed_size( VMARG );
FALCON_FUNC CellRenderer::get_visible( VMARG );
FALCON_FUNC CellRenderer::set_visible( VMARG );
FALCON_FUNC CellRenderer::get_sensitive( VMARG );
FALCON_FUNC CellRenderer::set_sensitive( VMARG );
FALCON_FUNC CellRenderer::get_alignment( VMARG );
FALCON_FUNC CellRenderer::set_alignment( VMARG );
FALCON_FUNC CellRenderer::get_padding( VMARG );
FALCON_FUNC CellRenderer::set_padding( VMARG );
#endif

} // Gtk
} // Falcon
