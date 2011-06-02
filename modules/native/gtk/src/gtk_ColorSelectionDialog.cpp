/**
 *  \file gtk_ColorSelectionDialog.cpp
 */

#include "gtk_ColorSelectionDialog.hpp"

#include "gtk_Widget.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ColorSelectionDialog::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ColorSelectionDialog = mod->addClass( "GtkColorSelectionDialog",
            &ColorSelectionDialog::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkDialog" ) );
    c_ColorSelectionDialog->getClassDef()->addInheritance( in );

    //c_ColorSelectionDialog->setWKS( true );
    //c_ColorSelectionDialog->getClassDef()->factory( &ColorSelectionDialog::factory );

#if GTK_CHECK_VERSION( 2, 14, 0 )
    mod->addClassMethod( c_ColorSelectionDialog, "get_color_selection",
                &ColorSelectionDialog::get_color_selection );
#endif
}


ColorSelectionDialog::ColorSelectionDialog( const Falcon::CoreClass* gen,
            const GtkColorSelectionDialog* dlg )
    :
    Gtk::CoreGObject( gen, (GObject*) dlg )
{}


Falcon::CoreObject* ColorSelectionDialog::factory( const Falcon::CoreClass* gen, void* dlg, bool )
{
    return new ColorSelectionDialog( gen, (GtkColorSelectionDialog*) dlg );
}


/*#
    @class GtkColorSelectionDialog
    @brief A standard dialog box for selecting a color
    @param title a string containing the title text for the dialog (or nil).

    The GtkColorSelectionDialog provides a standard dialog which allows the user
    to select a color much like the GtkFileSelection provides a standard dialog
    for file selection.

    [...]

 */
FALCON_FUNC ColorSelectionDialog::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    char* title = args.getCString( 0, false );
    if ( !title )
        title = (char*) "";
    MYSELF;
    GtkWidget* dlg = gtk_color_selection_dialog_new( title );
    self->setObject( (GObject*) dlg );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_color_selection GtkColorSelectionDialog
    @brief Retrieves the GtkColorSelection widget embedded in the dialog.
    @return the embedded GtkColorSelection.
 */
FALCON_FUNC ColorSelectionDialog::get_color_selection( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_color_selection_dialog_get_color_selection(
            (GtkColorSelectionDialog*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    /* Todo: return a GtkColorSelection instead. */
}
#endif


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
