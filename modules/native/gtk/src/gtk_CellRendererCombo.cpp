/**
 *  \file gtk_CellRendererCombo.cpp
 */

#include "gtk_CellRendererCombo.hpp"

#include "gtk_TreeIter.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void CellRendererCombo::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_CellRendererCombo = mod->addClass( "GtkCellRendererCombo", &CellRendererCombo::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkCellRendererText" ) );
    c_CellRendererCombo->getClassDef()->addInheritance( in );

    //c_CellRendererCombo->setWKS( true );
    c_CellRendererCombo->getClassDef()->factory( &CellRendererCombo::factory );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    mod->addClassMethod( c_CellRendererCombo,
                         "signal_changed",
                         &CellRendererCombo::signal_changed );
#endif
}


CellRendererCombo::CellRendererCombo( const Falcon::CoreClass* gen, const GtkCellRendererCombo* renderer )
    :
    Gtk::CoreGObject( gen, (GObject*) renderer )
{}


Falcon::CoreObject* CellRendererCombo::factory( const Falcon::CoreClass* gen, void* renderer, bool )
{
    return new CellRendererCombo( gen, (GtkCellRendererCombo*) renderer );
}


/*#
    @class GtkCellRendererCombo
    @brief Renders a combobox in a cell

    GtkCellRendererCombo renders text in a cell like GtkCellRendererText from
    which it is derived. But while GtkCellRendererText offers a simple entry to
    edit the text, GtkCellRendererCombo offers a GtkComboBox or GtkComboBoxEntry
    widget to edit the text. The values to display in the combo box are taken
    from the tree model specified in the model property.

    The combo cell renderer takes care of adding a text cell renderer to the
    combo box and sets it to display the column specified by its text-column
    property. Further properties of the comnbo box can be set in a handler for
    the editing-started signal.

    Adjust how text is drawn using object properties. Object properties can be
    set globally (with g_object_set()). Also, with GtkTreeViewColumn, you can
    bind a property to a value in a GtkTreeModel. For example, you can bind the
    "text" property on the cell renderer to a string value in the model, thus
    rendering a different string in each row of the GtkTreeView.
 */
FALCON_FUNC CellRendererCombo::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setObject( (GObject*) gtk_cell_renderer_combo_new() );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method signal_changed GtkCellRendererCombo
    @brief This signal is emitted each time after the user selected an item in the combo box, either by using the mouse or the arrow keys.

    Contrary to GtkComboBox, GtkCellRendererCombo::changed is not emitted for
    changes made to a selected item in the entry. The argument new_iter
    corresponds to the newly selected item in the combo box and it is relative
    to the GtkTreeModel set via the model property on GtkCellRendererCombo.

    Note that as soon as you change the model displayed in the tree view, the
    tree view will immediately cease the editing operating. This means that you
    most probably want to refrain from changing the model until the combo cell
    renderer emits the edited or editing_canceled signal.
 */
FALCON_FUNC CellRendererCombo::signal_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "changed", (void*) &CellRendererCombo::on_changed, vm );
}


void CellRendererCombo::on_changed( GtkCellRendererCombo* obj, gchar* path,
                                    GtkTreeIter* titer, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "changed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkTreeIter" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_changed", it ) )
            {
                printf(
                "[GtkCellRendererCombo::on_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( UTF8String( path ) );
        vm->pushParam( new Gtk::TreeIter( wki->asClass(), titer ) );
        vm->callItem( it, 2 );
    }
    while ( iter.hasCurrent() );
}
#endif // GTK_CHECK_VERSION( 2, 14, 0 )


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
