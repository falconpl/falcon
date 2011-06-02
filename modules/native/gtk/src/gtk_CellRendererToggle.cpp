/**
 *  \file gtk_CellRendererToggle.cpp
 */

#include "gtk_CellRendererToggle.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererToggle::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererToggle = mod->addClass( "GtkCellRendererToggle", &CellRendererToggle::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRenderer" ) );
    c_CellRendererToggle->getClassDef()->addInheritance( in );

    //c_CellRendererToggle->setWKS( true );
    c_CellRendererToggle->getClassDef()->factory( &CellRendererToggle::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",     CellRendererToggle::signal_toggled },
    { "get_radio",          CellRendererToggle::get_radio },
    { "set_radio",          CellRendererToggle::set_radio },
    { "get_active",         CellRendererToggle::get_active },
    { "set_active",         CellRendererToggle::set_active },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "get_activatable",    CellRendererToggle::get_activatable },
    { "set_activatable",    CellRendererToggle::set_activatable },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_CellRendererToggle, meth->name, meth->cb );
}


CellRendererToggle::CellRendererToggle( const Falcon::CoreClass* gen, const GtkCellRendererToggle* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererToggle::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererToggle( gen, (GtkCellRendererToggle*) renderer );
}


/*#
    @class GtkCellRendererToggle
    @brief Renders a toggle button in a cell

    GtkCellRendererToggle renders a toggle button in a cell. The button is drawn
    as a radio- or checkbutton, depending on the radio property. When activated,
    it emits the toggled signal.

    djust rendering parameters using object properties. Object properties can be
    set globally (with g_object_set()). Also, with GtkTreeViewColumn, you can
    bind a property to a value in a GtkTreeModel. For example, you can bind the
    "active" property on the cell renderer to a boolean value in the model, thus
    causing the check button to reflect the state of the model.
 */
FALCON_FUNC CellRendererToggle::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_accel_new() );
}


/*#
    @method signal_toggled GtkCellRendererToggle
    @brief The toggled signal is emitted when the cell is toggled.
 */
FALCON_FUNC CellRendererToggle::signal_toggled( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggled", (void*) &CellRendererToggle::on_toggled, vm );
}


void CellRendererToggle::on_toggled( GtkCellRendererToggle* obj, gchar* path, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "toggled", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_toggled", it ) )
            {
                printf(
                "[GtkCellRendererToggle::on_toggled] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( path ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


/*#
    @method get_radio GtkCellRendererToggle
    @brief Returns whether we're rendering radio toggles rather than checkboxes.
    @return TRUE if we're rendering radio toggles rather than checkboxes
 */
FALCON_FUNC CellRendererToggle::get_radio( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_cell_renderer_toggle_get_radio( GET_CELLRENDERERTOGGLE( vm->self() ) ) );
}


/*#
    @method set_radio GtkCellRendererToggle
    @brief If radio is TRUE, the cell renderer renders a radio toggle (i.e. a toggle in a group of mutually-exclusive toggles).
    @param radio TRUE to make the toggle look like a radio button

    If FALSE, it renders a check toggle (a standalone boolean option). This can
    be set globally for the cell renderer, or changed just before rendering each
    cell in the model (for GtkTreeView, you set up a per-row setting using
    GtkTreeViewColumn to associate model columns with cell renderer properties).
 */
FALCON_FUNC CellRendererToggle::set_radio( VMARG )
{
    Item* i_rad = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_rad || !i_rad->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_toggle_set_radio( GET_CELLRENDERERTOGGLE( vm->self() ),
                                        (gboolean) i_rad->asBoolean() );
}


/*#
    @method get_active GtkCellRendererToggle
    @brief Returns whether the cell renderer is active.
    @return TRUE if the cell renderer is active.
 */
FALCON_FUNC CellRendererToggle::get_active( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_cell_renderer_toggle_get_active( GET_CELLRENDERERTOGGLE( vm->self() ) ) );
}


/*#
    @method set_active GtkCellRendererToggle
    @brief Activates or deactivates a cell renderer.
    @param active TRUE to activate the cell renderer
 */
FALCON_FUNC CellRendererToggle::set_active( VMARG )
{
    Item* i_act = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || !i_act->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_toggle_set_active( GET_CELLRENDERERTOGGLE( vm->self() ),
                                         (gboolean) i_act->asBoolean() );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method get_activatable GtkCellRendererToggle
    @brief Returns whether the cell renderer is activatable.
    @return TRUE if the cell renderer is activatable.
 */
FALCON_FUNC CellRendererToggle::get_activatable( VMARG )
{
    NO_ARGS
    vm->retval( (bool) gtk_cell_renderer_toggle_get_activatable( GET_CELLRENDERERTOGGLE( vm->self() ) ) );
}


/*#
    @method set_activatable GtkCellRendererToggle
    @brief Makes the cell renderer activatable.
    @param activatable TRUE to make the cell renderer activatable
 */
FALCON_FUNC CellRendererToggle::set_activatable( VMARG )
{
    Item* i_act = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || !i_act->isBoolean() )
        throw_inv_params( "B" );
#endif
    gtk_cell_renderer_toggle_set_activatable( GET_CELLRENDERERTOGGLE( vm->self() ),
                                              (gboolean) i_act->asBoolean() );
}
#endif // GTK_CHECK_VERSION( 2, 18, 0 )

} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
