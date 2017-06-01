/*!
 *  Copyright (c) 2015 by Contributors
 * \file ml_dmlc_mxnet_native_c_api.cc
 * \brief JNI function implementations
 */
#include "ml_dmlc_mxnet_init_native_c_api.h"  // generated by javah
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>

JNIEXPORT jint JNICALL
Java_ml_dmlc_mxnet_init_LibInfo_mxSymbolListAtomicSymbolCreators(
    JNIEnv *env, jobject obj, jobject symbolList) {
    mx_uint outSize;
    AtomicSymbolCreator *outArray;
    int ret = MXSymbolListAtomicSymbolCreators(&outSize, &outArray);

    jclass longCls = env->FindClass("java/lang/Long");
    jmethodID longConst = env->GetMethodID(longCls, "<init>", "(J)V");

    jclass listCls = env->FindClass("scala/collection/mutable/ListBuffer");
    jmethodID listAppend = env->GetMethodID(
        listCls, "$plus$eq",
        "(Ljava/lang/Object;)Lscala/collection/mutable/ListBuffer;");

    for (size_t i = 0; i < outSize; ++i) {
        env->CallObjectMethod(symbolList, listAppend,
                              env->NewObject(longCls, longConst, outArray[i]));
    }

    return ret;
}

JNIEXPORT jint JNICALL
Java_ml_dmlc_mxnet_init_LibInfo_mxSymbolGetAtomicSymbolInfo(
    JNIEnv *env, jobject obj, jlong symbolPtr, jobject name, jobject desc,
    jobject numArgs, jobject argNames, jobject argTypes, jobject argDescs,
    jobject keyVarNumArgs) {
    const char *cName;
    const char *cDesc;
    mx_uint cNumArgs;
    const char **cArgNames;
    const char **cArgTypes;
    const char **cArgDescs;
    const char *cKeyVarNumArgs;

    int ret = MXSymbolGetAtomicSymbolInfo(
        reinterpret_cast<AtomicSymbolCreator>(symbolPtr), &cName, &cDesc,
        &cNumArgs, &cArgNames, &cArgTypes, &cArgDescs, &cKeyVarNumArgs);

    jclass refIntClass = env->FindClass("ml/dmlc/mxnet/init/Base$RefInt");
    jfieldID valueInt = env->GetFieldID(refIntClass, "value", "I");

    jclass refStringClass = env->FindClass("ml/dmlc/mxnet/init/Base$RefString");
    jfieldID valueStr =
        env->GetFieldID(refStringClass, "value", "Ljava/lang/String;");

    // scala.collection.mutable.ListBuffer append method
    jclass listClass = env->FindClass("scala/collection/mutable/ListBuffer");
    jmethodID listAppend = env->GetMethodID(
        listClass, "$plus$eq",
        "(Ljava/lang/Object;)Lscala/collection/mutable/ListBuffer;");

    env->SetObjectField(name, valueStr, env->NewStringUTF(cName));
    env->SetObjectField(desc, valueStr, env->NewStringUTF(cDesc));
    env->SetObjectField(keyVarNumArgs, valueStr,
                        env->NewStringUTF(cKeyVarNumArgs));
    env->SetIntField(numArgs, valueInt, static_cast<jint>(cNumArgs));
    for (size_t i = 0; i < cNumArgs; ++i) {
        env->CallObjectMethod(argNames, listAppend,
                              env->NewStringUTF(cArgNames[i]));
        env->CallObjectMethod(argTypes, listAppend,
                              env->NewStringUTF(cArgTypes[i]));
        env->CallObjectMethod(argDescs, listAppend,
                              env->NewStringUTF(cArgDescs[i]));
    }

    return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_mxnet_init_LibInfo_mxListAllOpNames(
    JNIEnv *env, jobject obj, jobject nameList) {
    mx_uint outSize;
    const char **outArray;
    int ret = MXListAllOpNames(&outSize, &outArray);

    jclass listCls = env->FindClass("scala/collection/mutable/ListBuffer");
    jmethodID listAppend = env->GetMethodID(
        listCls, "$plus$eq",
        "(Ljava/lang/Object;)Lscala/collection/mutable/ListBuffer;");
    for (size_t i = 0; i < outSize; ++i) {
        env->CallObjectMethod(nameList, listAppend,
                              env->NewStringUTF(outArray[i]));
    }

    return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_mxnet_init_LibInfo_nnGetOpHandle(
    JNIEnv *env, jobject obj, jstring jopname, jobject jhandle) {
    OpHandle handle;
    const char *opname = env->GetStringUTFChars(jopname, 0);
    int ret = NNGetOpHandle(opname, &handle);
    env->ReleaseStringUTFChars(jopname, opname);

    jclass refClass = env->FindClass("ml/dmlc/mxnet/init/Base$RefLong");
    jfieldID refFid = env->GetFieldID(refClass, "value", "J");
    env->SetLongField(jhandle, refFid, reinterpret_cast<jlong>(handle));

    return ret;
}
