/**
 *  \file gtk_FileChooserDialog.cpp
 */

#include "gtk_FileChooserDialog.hpp"

#include "gtk_FileChooser.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void FileChooserDialog::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_FileChooserDialog = mod->addClass( "GtkFileChooserDialog",
            &FileChooserDialog::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkDialog" ) );
    c_FileChooserDialog->getClassDef()->addInheritance( in );

    //c_FileChooserDialog->setWKS( true );
    //c_FileChooserDialog->getClassDef()->factory( &FileChooserDialog::factory );

#if 0
    mod->addClassMethod( c_FileChooserDialog, "get_color_selection",
                &FileChooserDialog::get_color_selection );
#endif

    /*
     *  implements GtkFileChooser
     */
    Gtk::FileChooser::clsInit( mod, c_FileChooserDialog );
}


FileChooserDialog::FileChooserDialog( const Falcon::CoreClass* gen,
            const GtkFileChooserDialog* dlg )
    :
    Gtk::CoreGObject( gen, (GObject*) dlg )
{}


Falcon::CoreObject* FileChooserDialog::factory( const Falcon::CoreClass* gen, void* dlg, bool )
{
    return new FileChooserDialog( gen, (GtkFileChooserDialog*) dlg );
}


/*#
    @class GtkFileChooserDialog
    @brief A file chooser dialog, suitable for "File/Open" or "File/Save" commands
    @param title Title of the dialog, or nil.
    @param parent Transient parent of the dialog (GtkWindow), or nil.
    @param action Open or save mode for the dialog (GtkFileChooserAction)

    The GtkFileChooserDialog provides a standard dialog which allows the user
    to select a color much like the GtkFileSelection provides a standard dialog
    for file selection.

    [...]

 */
FALCON_FUNC FileChooserDialog::init( VMARG )
{
    const char* spec = "[S,GtkWindow,I]";
    Gtk::ArgCheck1 args( vm, spec );

    char* title = args.getCString( 0, false );
    CoreGObject* o_parent = args.getCoreGObject( 1, false );
    int action = args.getInteger( 2, false );

#ifndef NO_PARAMETER_CHECK
    if ( o_parent && !CoreObject_IS_DERIVED( o_parent, GtkWindow ) )
        throw_inv_params( spec );
#endif
    GtkWindow* win = o_parent ? (GtkWindow*) o_parent->getGObject() : NULL;
    GtkWidget* dlg = gtk_file_chooser_dialog_new(
            title, win, (GtkFileChooserAction) action, NULL, NULL );
    MYSELF;
    self->setGObject( (GObject*) dlg );
}


} // Gtk
} // Falcon
