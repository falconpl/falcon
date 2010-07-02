/**
 *  \file gtk_CellEditable.cpp
 */

#include "gtk_CellEditable.hpp"

#include "gdk_Event.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellEditable::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellEditable = mod->addClass( "%GtkCellEditable" );

    c_CellEditable->setWKS( true );
    c_CellEditable->getClassDef()->factory( &CellEditable::factory );

    CellEditable::clsInit( mod, c_CellEditable );
}


/**
 *  \brief interface loader
 */
void CellEditable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "signal_editing_done",    &CellEditable::signal_editing_done },
    { "signal_remove_widget",   &CellEditable::signal_remove_widget },
    { "start_editing",          &CellEditable::start_editing },
    { "editing_done",           &CellEditable::editing_done },
    { "remove_widget",          &CellEditable::remove_widget },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


CellEditable::CellEditable( const Falcon::CoreClass* gen, const GtkCellEditable* editable )
    :
    Gtk::CoreGObject( gen, (GObject*) editable )
{}


Falcon::CoreObject* CellEditable::factory( const Falcon::CoreClass* gen, void* editable, bool )
{
    return new CellEditable( gen, (GtkCellEditable*) editable );
}


/*#
    @class GtkCellEditable
    @brief Interface for widgets which can are used for editing cells

    The GtkCellEditable interface must be implemented for widgets to be usable
    when editing the contents of a GtkTreeView cell.
 */


/*#
    @method signal_editing_done GtkCellEditable
    @brief This signal is a sign for the cell renderer to update its value from the cell_editable.

    Implementations of GtkCellEditable are responsible for emitting this signal
    when they are done editing, e.g. GtkEntry is emitting it when the user
    presses Enter.

    gtk_cell_editable_editing_done() is a convenience method for emitting
    GtkCellEditable::editing-done.
 */
FALCON_FUNC CellEditable::signal_editing_done( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "editing_done", (void*) &CellEditable::on_editing_done, vm );
}


void CellEditable::on_editing_done( GtkCellEditable* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "editing_done", "on_editing_done", (VMachine*)_vm );
}


/*#
    @method signal_remove_widget GtkCellEditable
    @brief This signal is meant to indicate that the cell is finished editing, and the widget may now be destroyed.

    Implementations of GtkCellEditable are responsible for emitting this signal
    when they are done editing. It must be emitted after the "editing-done"
    signal, to give the cell renderer a chance to update the cell's value before
    the widget is removed.

    gtk_cell_editable_remove_widget() is a convenience method for emitting
    GtkCellEditable::remove-widget.
 */
FALCON_FUNC CellEditable::signal_remove_widget( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "remove_widget", (void*) &CellEditable::on_remove_widget, vm );
}


void CellEditable::on_remove_widget( GtkCellEditable* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "remove_widget", "on_remove_widget", (VMachine*)_vm );
}


/*#
    @method start_editing GtkCellEditable
    @brief Begins editing on a cell_editable.
    @param event A GdkEvent, or NULL.

    event is the GdkEvent that began the editing process. It may be NULL, in
    the instance that editing was initiated through programatic means.
 */
FALCON_FUNC CellEditable::start_editing( VMARG )
{
    Item* i_ev = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_ev || !( i_ev->isNil() || ( i_ev->isObject()
        && IS_DERIVED( i_ev, GdkEvent ) ) ) )
        throw_inv_params( "[GdkEvent]" );
#endif
    GdkEvent* ev = i_ev->isNil() ? NULL : GET_EVENT( *i_ev );
    MYSELF;
    GET_OBJ( self );
    gtk_cell_editable_start_editing( (GtkCellEditable*)_obj, ev );
}


/*#
    @method editing_done GtkCellEditable
    @brief Emits the "editing-done" signal.
 */
FALCON_FUNC CellEditable::editing_done( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_cell_editable_editing_done( (GtkCellEditable*)_obj );
}


/*#
    @method remove_widget GtkCellEditable
    @brief Emits the "remove-widget" signal.
 */
FALCON_FUNC CellEditable::remove_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_cell_editable_remove_widget( (GtkCellEditable*)_obj );
}


} // Gtk
} // Falcon
