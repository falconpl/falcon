/**
 *  \file gtk_MessageDialog.cpp
 */

#include "gtk_MessageDialog.hpp"

#include "gtk_Widget.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void MessageDialog::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_MessageDialog = mod->addClass( "GtkMessageDialog", &MessageDialog::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkDialog" ) );
    c_MessageDialog->getClassDef()->addInheritance( in );

    c_MessageDialog->setWKS( true );
    c_MessageDialog->getClassDef()->factory( &MessageDialog::factory );

    Gtk::MethodTab methods[] =
    {
    { "new_with_markup",        &MessageDialog::new_with_markup },
    { "set_markup",             &MessageDialog::set_markup },
    { "set_image",              &MessageDialog::set_image },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_image",              &MessageDialog::get_image },
#endif
    { "set_secondary_text",     &MessageDialog::set_secondary_text },
    { "set_secondary_markup",   &MessageDialog::set_secondary_markup },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_MessageDialog, meth->name, meth->cb );
}


MessageDialog::MessageDialog( const Falcon::CoreClass* gen, const GtkMessageDialog* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* MessageDialog::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new MessageDialog( gen, (GtkMessageDialog*) btn );
}


/*#
    @class GtkMessageDialog
    @brief A convenient message window
    @param parent transient parent (GtkWindow), or nil for none.
    @param flags flags (GtkDialogFlags)
    @param type type of message (GtkMessageType)
    @param buttons set of buttons to use (GtkButtonsType)
    @param message message string, or nil.

    GtkMessageDialog presents a dialog with an image representing the type of
    message (Error, Question, etc.) alongside some message text. It's simply
    a convenience widget; you could construct the equivalent of
    GtkMessageDialog from GtkDialog without too much effort, but GtkMessageDialog
    saves typing.

    The easiest way to do a modal message dialog is to use gtk_dialog_run(),
    though you can also pass in the GTK_DIALOG_MODAL flag, gtk_dialog_run()
    automatically makes the dialog modal and waits for the user to respond
    to it. gtk_dialog_run() returns when any dialog button is clicked.
 */
FALCON_FUNC MessageDialog::init( VMARG )
{
    const char* spec = "GtkWindow,GtkDialogFlags,GtkMessageType,GtkButtonsType,S";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_parent = args.getCoreGObject( 0, false );
    int flags = args.getInteger( 1 );
    int type = args.getInteger( 2 );
    int buttons = args.getInteger( 3 );
    const gchar* msg = args.getCString( 4, false );
#ifndef NO_PARAMETER_CHECK
    if ( o_parent && !CoreObject_IS_DERIVED( o_parent, GtkWindow ) )
        throw_inv_params( spec );
#endif
    GtkWindow* parent = o_parent ? (GtkWindow*) o_parent->getObject() : NULL;
    GtkWidget* wdt = gtk_message_dialog_new( parent,
        (GtkDialogFlags) flags, (GtkMessageType) type, (GtkButtonsType) buttons,
        "%s", msg ); // can emit a warning, safely ignore.
    MYSELF;
    self->setObject( (GObject*) wdt );
}


/*#
    @method new_with_markup GtkMessageDialog
    @brief Creates a new message dialog.
    @param parent transient parent (GtkWindow), or nil for none.
    @param flags flags (GtkDialogFlags)
    @param type type of message (GtkMessageType)
    @param buttons set of buttons to use (GtkButtonsType)
    @param message message string, or nil.
    @return a new GtkMessageDialog.

    Creates a new message dialog, which is a simple dialog with an icon indicating
    the dialog type (error, warning, etc.) and some text which is marked up with
    the Pango text markup language. When the user clicks a button a "response"
    signal is emitted with response IDs from GtkResponseType. See GtkDialog for more details.

    [...]
 */
FALCON_FUNC MessageDialog::new_with_markup( VMARG )
{
    const char* spec = "GtkWindow,GtkDialogFlags,GtkMessageType,GtkButtonsType,S";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_parent = args.getCoreGObject( 0, false );
    int flags = args.getInteger( 1 );
    int type = args.getInteger( 2 );
    int buttons = args.getInteger( 3 );
    const gchar* msg = args.getCString( 4, false );
#ifndef NO_PARAMETER_CHECK
    if ( o_parent && !CoreObject_IS_DERIVED( o_parent, GtkWindow ) )
        throw_inv_params( spec );
#endif
    GtkWindow* parent = o_parent ? (GtkWindow*) o_parent->getObject() : NULL;
    GtkWidget* wdt = gtk_message_dialog_new_with_markup( parent,
        (GtkDialogFlags) flags, (GtkMessageType) type, (GtkButtonsType) buttons,
        "%s", msg ); // can emit a warning, safely ignore.
    vm->retval( new Gtk::MessageDialog(
            vm->findWKI( "GtkMessageDialog" )->asClass(), (GtkMessageDialog*) wdt ) );
}


/*#
    @method set_markup GtkMessageDialog
    @brief Sets the text of the message dialog to be str, which is marked up with the Pango text markup language.
    @param markup string
 */
FALCON_FUNC MessageDialog::set_markup( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* s = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_message_dialog_set_markup( (GtkMessageDialog*)_obj, s );
}


/*#
    @method set_image GtkMessageDialog
    @brief Sets the dialog's image to image
    @param the image (GtkWidget)
 */
FALCON_FUNC MessageDialog::set_image( VMARG )
{
    Item* i_img = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_img || i_img->isNil() || !i_img->isObject()
        || !IS_DERIVED( i_img, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* img = (GtkWidget*) COREGOBJECT( i_img )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_message_dialog_set_image( (GtkMessageDialog*)_obj, img );
}


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_image GtkMessageDialog
    @brief Gets the dialog's image.
    @return the dialog's image (GtkWidget)
 */
FALCON_FUNC MessageDialog::get_image( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* img = gtk_message_dialog_get_image( (GtkMessageDialog*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), img ) );
}
#endif


/*#
    @method set_secondary_text GtkMessageDialog
    @brief Sets the secondary text of the message dialog to be message.
    @param message string or nil.

    Note that setting a secondary text makes the primary text become bold, unless
    you have provided explicit markup.
 */
FALCON_FUNC MessageDialog::set_secondary_text( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* s = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_message_dialog_format_secondary_text( (GtkMessageDialog*)_obj, s, NULL );
}


/*#
    @method set_secondary_markup GtkMessageDialog
    @brief Sets the secondary text of the message dialog to be message, which is marked up with the Pango text markup language.

    Note that setting a secondary text makes the primary text become bold, unless
    you have provided explicit markup.

    Due to an oversight, this function does not escape special XML characters
    like gtk_message_dialog_new_with_markup() does. Thus, if the arguments
    may contain special XML characters, you should use g_markup_printf_escaped() to escape it.
 */
FALCON_FUNC MessageDialog::set_secondary_markup( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* s = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_message_dialog_format_secondary_markup( (GtkMessageDialog*)_obj, s, NULL );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
